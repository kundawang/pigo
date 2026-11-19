package pigo

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"time"
)

// Errors returned by Tracker.Update. Every error leaves the tracker state
// untouched, including its timestamp, so the caller can retry the same frame
// after fixing the input.
var (
	// ErrTimeRegression is returned when a frame timestamp is older than the
	// previously accepted frame.
	ErrTimeRegression = errors.New("pigo: frame timestamp regressed")
	// ErrDuplicateTimestamp is returned when a frame reuses the previously
	// accepted timestamp.
	ErrDuplicateTimestamp = errors.New("pigo: duplicate frame timestamp")
	// ErrInvalidFrameSize is returned when only one frame dimension is given,
	// or when a dimension is negative.
	ErrInvalidFrameSize = errors.New("pigo: invalid frame size")
	// ErrInvalidDetection is returned for a rectangle with non-positive or
	// non-finite dimensions, non-finite coordinates, or a confidence outside
	// [0, 1].
	ErrInvalidDetection = errors.New("pigo: invalid track detection")
	// ErrTooManyDetections is returned when a frame carries more detections
	// than TrackerConfig.MaxDetections.
	ErrTooManyDetections = errors.New("pigo: too many detections")
	// ErrTrackLimitExceeded is returned when creating tracks for the unmatched
	// detections would exceed TrackerConfig.MaxTracks.
	ErrTrackLimitExceeded = errors.New("pigo: track limit exceeded")
	// ErrInvalidTrackerConfig is returned by NewTracker for an unusable
	// configuration.
	ErrInvalidTrackerConfig = errors.New("pigo: invalid tracker config")
)

// MaxTrackerCandidates is the hard upper bound accepted for both
// TrackerConfig.MaxDetections and TrackerConfig.MaxTracks. It bounds the
// O(n^3) global assignment work and its O(n^2) scratch memory per frame.
const MaxTrackerCandidates = 256

const (
	// DefaultMatchThreshold is the default maximum assignment cost.
	DefaultMatchThreshold = 1.0
	// DefaultMaxMissedFrames is the default number of consecutive frames a
	// track may be missing before it is dropped.
	DefaultMaxMissedFrames = 5
	// DefaultMaxDetections is the default per-frame detection cap.
	DefaultMaxDetections = 64
	// DefaultMaxTracks is the default cap on concurrently tracked objects.
	DefaultMaxTracks = 64
	// DefaultPredictionHorizon is the default maximum time a track's motion is
	// extrapolated for matching and coasting output.
	DefaultPredictionHorizon = time.Second
	// DefaultVelocitySmoothing is the default weight of the previous velocity
	// estimate when incorporating a new measurement.
	DefaultVelocitySmoothing = 0.5

	trackConfidenceSmoothing = 0.5
	trackConfidenceDecay     = 0.8
	trackMinSize             = 1e-6
	// rejectCostMargin is added to MatchThreshold to build the cost used for
	// pairs that must not match. Using one constant above the threshold makes
	// the assignment solver treat all unusable pairs equally, so they never
	// distort which usable pairs are chosen.
	rejectCostMargin = 1.0

	maxMissedDurationLimit    = 24 * time.Hour
	maxPredictionHorizonLimit = 24 * time.Hour
)

// Rect is an axis-aligned bounding box in pixel coordinates. (X, Y) is the
// top-left corner and Width/Height are positive extents.
type Rect struct {
	X      float64
	Y      float64
	Width  float64
	Height float64
}

// Valid reports whether r has finite coordinates and positive, finite
// dimensions. Detections with an invalid rectangle are rejected by
// Tracker.Update.
func (r Rect) Valid() bool {
	values := [...]float64{r.X, r.Y, r.Width, r.Height}
	for _, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	if r.Width <= 0 || r.Height <= 0 {
		return false
	}
	// The derived corners must be finite as well, so center arithmetic used by
	// the tracker cannot overflow.
	return !math.IsInf(r.X+r.Width, 0) && !math.IsInf(r.Y+r.Height, 0)
}

func (r Rect) center() (float64, float64) {
	return r.X + r.Width/2, r.Y + r.Height/2
}

// DetectionRect converts a cascade detection, whose Row/Col denote the window
// center and whose Scale denotes its width and height, into a Rect. Non-positive
// scales produce an invalid Rect.
func DetectionRect(d Detection) Rect {
	return Rect{
		X:      float64(d.Col) - float64(d.Scale)/2,
		Y:      float64(d.Row) - float64(d.Scale)/2,
		Width:  float64(d.Scale),
		Height: float64(d.Scale),
	}
}

// TrackDetection is one detected object in a frame. Confidence must be in
// [0, 1]; Detections produced by pigo's cascade use an unbounded score, so callers
// should normalize it before constructing TrackDetection.
type TrackDetection struct {
	Rect       Rect
	Confidence float64
}

// Frame is the input of Tracker.Update: the detections of one video frame and
// the frame timestamp. Timestamps must increase strictly between accepted
// frames.
//
// Width and Height are optional. When both are zero the tracker keeps its last
// known resolution. When both are positive and differ from the previous known
// resolution, all track positions, sizes and velocities are rescaled
// proportionally before matching, so a resolution switch does not break IDs.
type Frame struct {
	Timestamp  time.Time
	Width      int
	Height     int
	Detections []TrackDetection
}

// TrackerConfig controls matching and track lifetime. Use
// DefaultTrackerConfig and adjust the fields that matter; zero values are not
// treated as defaults and usually fail validation.
type TrackerConfig struct {
	// MatchThreshold is the largest assignment cost accepted as a match. Cost
	// is the predicted distance divided by the predicted object size, plus a
	// size-difference term (also normalized by size), so it is resolution
	// independent. Must be positive and finite.
	MatchThreshold float64
	// MaxMissedFrames is the number of consecutive accepted frames a track may
	// go unmatched and still be kept (and re-acquired). A track that would miss
	// for the (MaxMissedFrames+1)-th consecutive frame is dropped in that frame:
	// its ID is added to FrameResult.Exited, it is not reported, and its slot is
	// immediately free for a new track. Zero means a track must be matched on
	// every frame. Values are clamped to MaxTrackerCandidates by NewTracker.
	MaxMissedFrames int
	// MaxMissedDuration is a wall-clock grace period. A track whose last sighting
	// is older than this is dropped before matching, even if a detection now
	// overlaps it. Zero disables the wall-clock limit.
	MaxMissedDuration time.Duration
	// MaxDetections caps the detections accepted per frame. Exceeding it makes
	// Update return ErrTooManyDetections without changing state.
	MaxDetections int
	// MaxTracks caps the number of live tracks, including coasting tracks.
	// Exceeding it makes Update return ErrTrackLimitExceeded without changing
	// state.
	MaxTracks int
	// PredictionHorizon limits how far into the future a track's velocity is
	// extrapolated. It also bounds how long a coasting track's reported
	// rectangle keeps moving before it is frozen. NewTracker clamps it to
	// maxPredictionHorizonLimit so extrapolation cannot run away.
	PredictionHorizon time.Duration
	// VelocitySmoothing is the weight of the existing velocity in [0, 1) when a
	// new measurement is folded in. Zero uses the newest measurement only.
	VelocitySmoothing float64
}

// DefaultTrackerConfig returns a usable configuration for typical video rates.
func DefaultTrackerConfig() TrackerConfig {
	return TrackerConfig{
		MatchThreshold:    DefaultMatchThreshold,
		MaxMissedFrames:   DefaultMaxMissedFrames,
		MaxDetections:     DefaultMaxDetections,
		MaxTracks:         DefaultMaxTracks,
		PredictionHorizon: DefaultPredictionHorizon,
		VelocitySmoothing: DefaultVelocitySmoothing,
	}
}

func (c TrackerConfig) validate() error {
	if math.IsNaN(c.MatchThreshold) || math.IsInf(c.MatchThreshold, 0) || c.MatchThreshold <= 0 {
		return fmt.Errorf("%w: MatchThreshold must be positive and finite", ErrInvalidTrackerConfig)
	}
	if c.MaxMissedFrames < 0 {
		return fmt.Errorf("%w: MaxMissedFrames must not be negative", ErrInvalidTrackerConfig)
	}
	if c.MaxMissedDuration < 0 {
		return fmt.Errorf("%w: MaxMissedDuration must not be negative", ErrInvalidTrackerConfig)
	}
	if c.MaxDetections <= 0 || c.MaxDetections > MaxTrackerCandidates {
		return fmt.Errorf("%w: MaxDetections must be in [1, %d]", ErrInvalidTrackerConfig, MaxTrackerCandidates)
	}
	if c.MaxTracks <= 0 || c.MaxTracks > MaxTrackerCandidates {
		return fmt.Errorf("%w: MaxTracks must be in [1, %d]", ErrInvalidTrackerConfig, MaxTrackerCandidates)
	}
	if c.PredictionHorizon <= 0 {
		return fmt.Errorf("%w: PredictionHorizon must be positive", ErrInvalidTrackerConfig)
	}
	if math.IsNaN(c.VelocitySmoothing) || c.VelocitySmoothing < 0 || c.VelocitySmoothing >= 1 {
		return fmt.Errorf("%w: VelocitySmoothing must be in [0, 1)", ErrInvalidTrackerConfig)
	}
	return nil
}

// Track is a stable trajectory as reported by Tracker.Update. For a track that
// was matched in the current frame, Rect is the observed rectangle. For a
// coasting track it is the predicted rectangle at the frame timestamp.
type Track struct {
	ID         int
	Rect       Rect
	Confidence float64
	// Hits is the number of frames in which the track was matched.
	Hits int
	// Missed is the current streak of accepted frames without a match.
	Missed int
	// Age is the number of accepted frames since the track was created.
	Age int
	// LastSeen is the timestamp of the last matched frame.
	LastSeen time.Time
	// Coasting reports whether the track is currently unmatched.
	Coasting bool
}

// FrameResult is the outcome of an accepted frame.
type FrameResult struct {
	// Tracks contains every live track (matched and coasting), ordered by ID.
	// A track whose max missed count would be exceeded in this frame is already
	// gone and appears in Exited instead.
	Tracks []Track
	// Assignments is parallel to Frame.Detections and holds the ID of the track
	// that owns each detection. Every accepted detection is assigned: an
	// unmatched detection starts a new track.
	Assignments []int
	// NewTracks lists IDs created in this frame, in detection order.
	NewTracks []int
	// Exited lists IDs removed in this frame, in ascending order.
	Exited []int
}

// Tracker is a lightweight, deterministic multi-object tracker intended to
// give pigo's per-frame face detections a stable identity. It predicts each
// track's position and size from velocity and solves the whole frame's
// assignment globally, so crossing faces, zooming faces and newly appearing
// faces do not cause avoidable ID swaps.
//
// A Tracker is not safe for concurrent use.
type Tracker struct {
	cfg TrackerConfig

	tracks   []*trackState // invariant: ascending ID
	nextID   int
	lastTime time.Time
	hasTime  bool
	width    int
	height   int
}

type trackState struct {
	id         int
	rect       Rect
	confidence float64
	vx, vy     float64 // pixels per second
	vw, vh     float64 // size change per second
	hits       int
	missed     int
	age        int
	lastSeen   time.Time
}

// NewTracker validates cfg and returns a tracker with no tracks. Unbounded
// grace values are clamped: MaxMissedFrames to MaxTrackerCandidates and
// MaxMissedDuration to 24h, so a single stale track cannot live forever.
func NewTracker(cfg TrackerConfig) (*Tracker, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	cfg.MaxMissedFrames = min(cfg.MaxMissedFrames, MaxTrackerCandidates)
	if cfg.MaxMissedDuration > maxMissedDurationLimit {
		cfg.MaxMissedDuration = maxMissedDurationLimit
	}
	if cfg.PredictionHorizon > maxPredictionHorizonLimit {
		cfg.PredictionHorizon = maxPredictionHorizonLimit
	}
	return &Tracker{cfg: cfg}, nil
}

// Reset removes all tracks and forgets the last timestamp and resolution. IDs
// start again from 1. The configuration is kept.
func (t *Tracker) Reset() {
	t.tracks = nil
	t.nextID = 0
	t.lastTime = time.Time{}
	t.hasTime = false
	t.width = 0
	t.height = 0
}

// Update consumes one frame and returns the live tracks for that timestamp.
//
// Input validation happens before any state is modified. On error the tracker
// is exactly as it was before the call, including the clock, so the frame may
// be retried. In particular:
//   - a timestamp older than the previous one yields ErrTimeRegression;
//   - an equal timestamp yields ErrDuplicateTimestamp;
//   - a bad rectangle or confidence yields ErrInvalidDetection;
//   - presenting more than MaxDetections detections yields ErrTooManyDetections;
//   - needing more than MaxTracks live tracks yields ErrTrackLimitExceeded.
//
// All input checks, including the track cap, run before state is changed, so a
// rejected frame consumes neither time, nor IDs, nor track slots.
//
// Empty detections are valid: matching is skipped and every live track coasts
// for the frame. An empty frame is therefore a way to advance a clock during
// dropped or fully occluded frames without inventing detections.
func (t *Tracker) Update(f Frame) (FrameResult, error) {
	if err := t.validateFrame(f); err != nil {
		return FrameResult{}, err
	}

	// Work on copies so no error below can leave partially applied state.
	alive := make([]trackState, 0, len(t.tracks))
	exited := make([]int, 0)
	sx, sy := t.resizeFactors(f)
	for _, st := range t.tracks {
		snap := *st
		if sx != 1 || sy != 1 {
			snap.rect.X *= sx
			snap.rect.Y *= sy
			snap.rect.Width *= sx
			snap.rect.Height *= sy
			snap.vx *= sx
			snap.vy *= sy
			snap.vw *= sx
			snap.vh *= sy
		}
		// A track unseen for longer than the wall-clock grace cannot be
		// re-acquired by this frame, even if a detection now covers it.
		if t.cfg.MaxMissedDuration > 0 && f.Timestamp.Sub(snap.lastSeen) > t.cfg.MaxMissedDuration {
			exited = append(exited, snap.id)
			continue
		}
		alive = append(alive, snap)
	}

	trackMatch, detMatch := t.match(alive, f.Detections, f.Timestamp)

	// Tracks that miss for the last time in this frame are dropped right away,
	// so they do not consume track slots and a new face can take their place in
	// the same frame.
	lastMiss := make([]bool, len(alive))
	live := len(alive)
	newCount := 0
	for i := range alive {
		if trackMatch[i] < 0 && alive[i].missed+1 > t.cfg.MaxMissedFrames {
			lastMiss[i] = true
			live--
		}
	}
	for _, id := range detMatch {
		if id < 0 {
			newCount++
		}
	}
	if live+newCount > t.cfg.MaxTracks {
		return FrameResult{}, fmt.Errorf(
			"%w: %d live tracks plus %d new detections exceeds MaxTracks %d",
			ErrTrackLimitExceeded, live, newCount, t.cfg.MaxTracks,
		)
	}

	assignments := make([]int, len(f.Detections))
	for i := range assignments {
		assignments[i] = -1
	}
	newTracks := make([]int, 0, newCount)
	committed := make([]*trackState, 0, len(alive)+newCount)
	out := make([]Track, 0, len(alive)+newCount)

	for i := range alive {
		st := &alive[i]
		st.age++
		if lastMiss[i] {
			exited = append(exited, st.id)
			continue
		}
		if j := trackMatch[i]; j >= 0 {
			t.applyDetection(st, f.Detections[j], f.Timestamp)
			assignments[j] = st.id
			out = append(out, st.export(st.rect))
			committed = append(committed, st)
			continue
		}

		st.missed++
		st.confidence *= trackConfidenceDecay
		rep := t.predict(alive[i], f.Timestamp)
		// Predictions are not clamped by the detector, so keep their corners
		// finite: a wildly extrapolated velocity could overflow into Inf.
		if !rep.Valid() {
			rep = alive[i].rect
		}
		out = append(out, st.export(rep))
		committed = append(committed, st)
	}

	for j := range f.Detections {
		if detMatch[j] >= 0 {
			continue
		}
		det := f.Detections[j]
		id := t.nextID + 1
		st := &trackState{
			id:         id,
			rect:       det.Rect,
			confidence: det.Confidence,
			hits:       1,
			age:        1,
			lastSeen:   f.Timestamp,
		}
		t.nextID = id
		newTracks = append(newTracks, id)
		assignments[j] = id
		committed = append(committed, st)
		out = append(out, st.export(st.rect))
	}

	// New IDs are always larger than existing ones, but sorting keeps the
	// documented ordering explicit and independent of that invariant.
	sort.Slice(committed, func(i, j int) bool { return committed[i].id < committed[j].id })
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	sort.Ints(exited)

	t.tracks = committed
	t.lastTime = f.Timestamp
	t.hasTime = true
	if f.Width > 0 {
		t.width, t.height = f.Width, f.Height
	}

	return FrameResult{
		Tracks:      out,
		Assignments: assignments,
		NewTracks:   newTracks,
		Exited:      exited,
	}, nil
}

func (t *Tracker) validateFrame(f Frame) error {
	if t.hasTime {
		switch dt := f.Timestamp.Sub(t.lastTime); {
		case dt < 0:
			return fmt.Errorf("%w: got %s after %s", ErrTimeRegression, f.Timestamp, t.lastTime)
		case dt == 0:
			return fmt.Errorf("%w: %s", ErrDuplicateTimestamp, f.Timestamp)
		}
	}
	if f.Width < 0 || f.Height < 0 || (f.Width == 0) != (f.Height == 0) {
		return fmt.Errorf("%w: got %dx%d", ErrInvalidFrameSize, f.Width, f.Height)
	}
	if len(f.Detections) > t.cfg.MaxDetections {
		return fmt.Errorf(
			"%w: %d detections exceed MaxDetections %d",
			ErrTooManyDetections, len(f.Detections), t.cfg.MaxDetections,
		)
	}
	for i, d := range f.Detections {
		if !d.Rect.Valid() {
			return fmt.Errorf("%w at index %d: rectangle %+v", ErrInvalidDetection, i, d.Rect)
		}
		if math.IsNaN(d.Confidence) || math.IsInf(d.Confidence, 0) ||
			d.Confidence < 0 || d.Confidence > 1 {
			return fmt.Errorf(
				"%w at index %d: confidence %v outside [0, 1]",
				ErrInvalidDetection, i, d.Confidence,
			)
		}
	}
	return nil
}

func (t *Tracker) resizeFactors(f Frame) (float64, float64) {
	if f.Width == 0 || t.width == 0 || (f.Width == t.width && f.Height == t.height) {
		return 1, 1
	}
	return float64(f.Width) / float64(t.width), float64(f.Height) / float64(t.height)
}

// match solves the frame assignment globally. It returns, for every live
// track, the index of its matched detection and, for every detection, the
// index of its matched live track; -1 means no match.
func (t *Tracker) match(tracks []trackState, dets []TrackDetection, now time.Time) ([]int, []int) {
	trackMatch := make([]int, len(tracks))
	detMatch := make([]int, len(dets))
	for i := range trackMatch {
		trackMatch[i] = -1
	}
	for i := range detMatch {
		detMatch[i] = -1
	}
	if len(tracks) == 0 || len(dets) == 0 {
		return trackMatch, detMatch
	}

	n := max(len(tracks), len(dets))
	cost := make([][]float64, n)
	for i := range cost {
		cost[i] = make([]float64, n)
	}
	for i := 0; i < len(tracks); i++ {
		for j := 0; j < len(dets); j++ {
			c := t.matchCost(tracks[i], dets[j], now)
			// The negated comparison also catches NaN, so only finite costs
			// below the threshold reach the solver.
			if !(c <= t.cfg.MatchThreshold) {
				c = t.cfg.MatchThreshold + rejectCostMargin
			}
			cost[i][j] = c
		}
	}
	// Pairs whose cost exceeds MatchThreshold are replaced by one common cost
	// just above it. This keeps them unusable while treating them identically,
	// so they never distort which usable pairs the solver chooses; it also
	// replaces NaN/Inf costs, which keeps the solver free of NaN potentials.

	rows := solveAssignment(cost)
	for i := 0; i < len(tracks); i++ {
		j := rows[i]
		if j < 0 || j >= len(dets) || cost[i][j] > t.cfg.MatchThreshold || detMatch[j] >= 0 {
			continue
		}
		trackMatch[i] = j
		detMatch[j] = i
	}
	return trackMatch, detMatch
}

func (t *Tracker) matchCost(st trackState, d TrackDetection, now time.Time) float64 {
	pred := t.predict(st, now)
	pcx, pcy := pred.center()
	dcx, dcy := d.Rect.center()
	ref := (pred.Width + pred.Height) / 2
	if ref < trackMinSize {
		ref = trackMinSize
	}
	centerCost := math.Hypot(dcx-pcx, dcy-pcy) / ref
	sizeCost := 0.5 * (math.Abs(d.Rect.Width-pred.Width)/ref +
		math.Abs(d.Rect.Height-pred.Height)/ref)
	cost := centerCost + sizeCost
	if math.IsNaN(cost) || math.IsInf(cost, 0) {
		return math.Inf(1)
	}
	return cost
}

func (t *Tracker) predict(st trackState, now time.Time) Rect {
	dt := now.Sub(st.lastSeen).Seconds()
	if dt <= 0 {
		return st.rect
	}
	if horizon := t.cfg.PredictionHorizon.Seconds(); dt > horizon {
		dt = horizon
	}
	cx, cy := st.rect.center()
	w := st.rect.Width + st.vw*dt
	h := st.rect.Height + st.vh*dt
	if w < trackMinSize {
		w = trackMinSize
	}
	if h < trackMinSize {
		h = trackMinSize
	}
	cx += st.vx * dt
	cy += st.vy * dt
	pred := Rect{X: cx - w/2, Y: cy - h/2, Width: w, Height: h}
	if !pred.Valid() {
		return st.rect
	}
	return pred
}

func (t *Tracker) applyDetection(st *trackState, det TrackDetection, now time.Time) {
	dt := now.Sub(st.lastSeen).Seconds()
	if dt > 0 {
		oldCX, oldCY := st.rect.center()
		newCX, newCY := det.Rect.center()
		rvx := (newCX - oldCX) / dt
		rvy := (newCY - oldCY) / dt
		rvw := (det.Rect.Width - st.rect.Width) / dt
		rvh := (det.Rect.Height - st.rect.Height) / dt
		if finite(rvx) && finite(rvy) && finite(rvw) && finite(rvh) {
			if st.hits <= 1 {
				st.vx, st.vy, st.vw, st.vh = rvx, rvy, rvw, rvh
			} else {
				a := t.cfg.VelocitySmoothing
				st.vx = a*st.vx + (1-a)*rvx
				st.vy = a*st.vy + (1-a)*rvy
				st.vw = a*st.vw + (1-a)*rvw
				st.vh = a*st.vh + (1-a)*rvh
			}
		}
	}
	st.rect = det.Rect
	st.lastSeen = now
	st.confidence = trackConfidenceSmoothing*st.confidence +
		(1-trackConfidenceSmoothing)*det.Confidence
	st.confidence = min(max(st.confidence, 0), 1)
	st.hits++
	st.missed = 0
}

func (st *trackState) export(r Rect) Track {
	return Track{
		ID:         st.id,
		Rect:       r,
		Confidence: st.confidence,
		Hits:       st.hits,
		Missed:     st.missed,
		Age:        st.age,
		LastSeen:   st.lastSeen,
		Coasting:   st.missed > 0,
	}
}

func finite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// solveAssignment returns a minimum-cost perfect matching for a square cost
// matrix, using the shortest augmenting path Hungarian algorithm. Every row is
// assigned a distinct column. The result is deterministic: rows and columns are
// scanned in ascending order and ties keep the first candidate, so identical
// inputs always produce identical assignments. Costs must be finite.
func solveAssignment(cost [][]float64) []int {
	n := len(cost)
	assign := make([]int, n)
	for i := range assign {
		assign[i] = -1
	}
	if n == 0 {
		return assign
	}

	u := make([]float64, n+1)
	v := make([]float64, n+1)
	p := make([]int, n+1) // p[j]: row (1-based) assigned to column j
	way := make([]int, n+1)

	for i := 1; i <= n; i++ {
		p[0] = i
		j0 := 0
		minv := make([]float64, n+1)
		used := make([]bool, n+1)
		for j := 1; j <= n; j++ {
			minv[j] = math.Inf(1)
		}
		for {
			used[j0] = true
			i0 := p[j0]
			delta := math.Inf(1)
			j1 := 0
			for j := 1; j <= n; j++ {
				if used[j] {
					continue
				}
				cur := cost[i0-1][j-1] - u[i0] - v[j]
				if cur < minv[j] {
					minv[j] = cur
					way[j] = j0
				}
				if minv[j] < delta {
					delta = minv[j]
					j1 = j
				}
			}
			for j := 0; j <= n; j++ {
				if used[j] {
					u[p[j]] += delta
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}
			j0 = j1
			if p[j0] == 0 {
				break
			}
		}
		for {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0 {
				break
			}
		}
	}

	for j := 1; j <= n; j++ {
		if p[j] != 0 {
			assign[p[j]-1] = j - 1
		}
	}
	return assign
}
