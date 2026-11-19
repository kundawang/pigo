package pigo_test

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	pigo "github.com/esimov/pigo/pkg"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var trackBase = time.Date(2026, 11, 20, 12, 0, 0, 0, time.UTC)

// frameAt builds a frame timestamped n video frames (at fps) after trackBase.
func frameAt(n int, dets ...pigo.TrackDetection) pigo.Frame {
	return pigo.Frame{
		Timestamp:  trackBase.Add(time.Duration(n) * time.Second / 30),
		Detections: dets,
	}
}

func det(x, y, w, h, conf float64) pigo.TrackDetection {
	return pigo.TrackDetection{Rect: pigo.Rect{X: x, Y: y, Width: w, Height: h}, Confidence: conf}
}

func newTestTracker(t *testing.T, modify func(*pigo.TrackerConfig)) *pigo.Tracker {
	t.Helper()
	cfg := pigo.DefaultTrackerConfig()
	if modify != nil {
		modify(&cfg)
	}
	tr, err := pigo.NewTracker(cfg)
	require.NoError(t, err)
	return tr
}

func mustUpdate(t *testing.T, tr *pigo.Tracker, f pigo.Frame) pigo.FrameResult {
	t.Helper()
	res, err := tr.Update(f)
	require.NoError(t, err)
	return res
}

func trackByID(res pigo.FrameResult, id int) (pigo.Track, bool) {
	for _, tr := range res.Tracks {
		if tr.ID == id {
			return tr, true
		}
	}
	return pigo.Track{}, false
}

func trackIDs(res pigo.FrameResult) []int {
	ids := make([]int, len(res.Tracks))
	for i, tr := range res.Tracks {
		ids[i] = tr.ID
	}
	return ids
}

func TestTracker_NewTrackerValidatesConfig(t *testing.T) {
	valid := pigo.DefaultTrackerConfig()
	_, err := pigo.NewTracker(valid)
	require.NoError(t, err)

	zero := pigo.TrackerConfig{}
	_, err = pigo.NewTracker(zero)
	assert.ErrorIs(t, err, pigo.ErrInvalidTrackerConfig, "zero config must not silently use defaults")

	cases := map[string]func(*pigo.TrackerConfig){
		"zero threshold":     func(c *pigo.TrackerConfig) { c.MatchThreshold = 0 },
		"negative threshold": func(c *pigo.TrackerConfig) { c.MatchThreshold = -1 },
		"NaN threshold":      func(c *pigo.TrackerConfig) { c.MatchThreshold = math.NaN() },
		"Inf threshold":      func(c *pigo.TrackerConfig) { c.MatchThreshold = math.Inf(1) },
		"negative missed":    func(c *pigo.TrackerConfig) { c.MaxMissedFrames = -1 },
		"negative duration":  func(c *pigo.TrackerConfig) { c.MaxMissedDuration = -time.Second },
		"zero max dets":      func(c *pigo.TrackerConfig) { c.MaxDetections = 0 },
		"huge max dets":      func(c *pigo.TrackerConfig) { c.MaxDetections = pigo.MaxTrackerCandidates + 1 },
		"zero max tracks":    func(c *pigo.TrackerConfig) { c.MaxTracks = 0 },
		"huge max tracks":    func(c *pigo.TrackerConfig) { c.MaxTracks = pigo.MaxTrackerCandidates + 1 },
		"zero horizon":       func(c *pigo.TrackerConfig) { c.PredictionHorizon = 0 },
		"negative smoothing": func(c *pigo.TrackerConfig) { c.VelocitySmoothing = -0.1 },
		"smoothing one":      func(c *pigo.TrackerConfig) { c.VelocitySmoothing = 1 },
		"NaN smoothing":      func(c *pigo.TrackerConfig) { c.VelocitySmoothing = math.NaN() },
	}
	for name, mutate := range cases {
		cfg := pigo.DefaultTrackerConfig()
		mutate(&cfg)
		_, err := pigo.NewTracker(cfg)
		assert.ErrorIs(t, err, pigo.ErrInvalidTrackerConfig, name)
	}
}

func TestTracker_EmptyDetectionsAndFirstFrame(t *testing.T) {
	tr := newTestTracker(t, nil)

	res := mustUpdate(t, tr, frameAt(0))
	assert.Empty(t, res.Tracks)
	assert.Empty(t, res.Assignments)
	assert.Empty(t, res.NewTracks)
	assert.Empty(t, res.Exited)

	res = mustUpdate(t, tr, frameAt(1, det(10, 10, 50, 50, 0.9), det(200, 100, 40, 40, 0.5)))
	assert.Equal(t, []int{1, 2}, res.NewTracks)
	assert.Equal(t, []int{1, 2}, res.Assignments, "assignments are parallel to detections")
	assert.Equal(t, []int{1, 2}, trackIDs(res))
	for _, tk := range res.Tracks {
		assert.True(t, tk.Rect.Valid())
		assert.False(t, tk.Coasting)
		assert.Equal(t, 1, tk.Hits)
		assert.Equal(t, 1, tk.Age)
	}
}

func TestTracker_StableIDWhileMoving(t *testing.T) {
	tr := newTestTracker(t, nil)

	for i := 0; i < 20; i++ {
		res := mustUpdate(t, tr, frameAt(i, det(float64(100+3*i), float64(80+2*i), 60, 60, 0.9)))
		require.Len(t, res.Tracks, 1, "frame %d", i)
		assert.Equal(t, 1, res.Assignments[0], "frame %d", i)
		tk := res.Tracks[0]
		assert.Equal(t, 1, tk.ID)
		assert.False(t, tk.Coasting)
		assert.InDelta(t, 100+3*i, tk.Rect.X, 0.001)
		assert.InDelta(t, 80+2*i, tk.Rect.Y, 0.001)
	}
}

func TestTracker_TwoPeopleCrossKeepIDs(t *testing.T) {
	tr := newTestTracker(t, nil)

	// A moves left to right, B moves right to left. They overlap in the middle
	// of the sequence and have identical size and confidence, so only the
	// motion prediction and the global assignment can keep them apart.
	const frames = 24
	const speed = 15.0
	const size = 50.0
	leftStart, rightStart := 60.0, 60.0+speed*frames

	for i := 0; i < frames; i++ {
		aX := leftStart + speed*float64(i)
		bX := rightStart - speed*float64(i)
		// Keep a stable detection order independent of positions so the tracker
		// cannot rely on slice order.
		detsA := det(aX, 100, size, size, 0.8)
		detsB := det(bX, 105, size, size, 0.8)
		res := mustUpdate(t, tr, frameAt(i, detsA, detsB))

		require.Len(t, res.Tracks, 2, "frame %d", i)
		assert.Equal(t, 1, res.Assignments[0], "A keeps ID 1 on frame %d", i)
		assert.Equal(t, 2, res.Assignments[1], "B keeps ID 2 on frame %d", i)
	}
	// At the crossing point both detections overlap in X; confirm it really
	// happened so the test is meaningful.
	cross := frames / 2
	aX := leftStart + speed*float64(cross)
	bX := rightStart - speed*float64(cross)
	require.Less(t, math.Abs(aX-bX), size, "the two faces must overlap at the crossing frame")
}

func TestTracker_ThreeFrameOcclusionKeepsID(t *testing.T) {
	tr := newTestTracker(t, func(c *pigo.TrackerConfig) { c.MaxMissedFrames = 3 })

	mustUpdate(t, tr, frameAt(0, det(100, 100, 60, 60, 0.9)))
	mustUpdate(t, tr, frameAt(1, det(104, 100, 60, 60, 0.9)))

	// Three frames without any detection: the track coasts, keeps its ID and
	// its prediction continues along the learned velocity.
	for i := 2; i <= 4; i++ {
		res := mustUpdate(t, tr, frameAt(i))
		require.Len(t, res.Tracks, 1, "frame %d", i)
		assert.Empty(t, res.Exited, "grace period must not exit on frame %d", i)
		tk := res.Tracks[0]
		assert.Equal(t, 1, tk.ID)
		assert.True(t, tk.Coasting)
		assert.Equal(t, i-1, tk.Missed)
		assert.Greater(t, tk.Rect.X, 104.0, "prediction must keep moving right on frame %d", i)
	}

	// The face reappears: it keeps its ID instead of starting a new track.
	res := mustUpdate(t, tr, frameAt(5, det(118, 100, 60, 60, 0.9)))
	require.Len(t, res.Tracks, 1)
	assert.Empty(t, res.NewTracks)
	assert.Equal(t, []int{1}, res.Assignments)
	assert.False(t, res.Tracks[0].Coasting)
	assert.Equal(t, 3, res.Tracks[0].Hits)

	// Reappear at frame 5: the track keeps its ID and resets its miss streak.
	// Disappear again for the full grace plus one frame: it exits exactly then.
	res = mustUpdate(t, tr, frameAt(6))
	assert.Empty(t, res.Exited, "first miss after re-acquisition")
	res = mustUpdate(t, tr, frameAt(7))
	assert.Empty(t, res.Exited, "second miss")
	res = mustUpdate(t, tr, frameAt(8))
	assert.Empty(t, res.Exited, "third miss is still within the grace")
	res = mustUpdate(t, tr, frameAt(9))
	assert.Empty(t, res.Tracks)
	assert.Equal(t, []int{1}, res.Exited)
}

func TestTracker_ExitsAfterGraceAndRejectsStaleReappearance(t *testing.T) {
	tr := newTestTracker(t, func(c *pigo.TrackerConfig) {
		c.MaxMissedFrames = 2
		c.MaxMissedDuration = time.Second
	})

	mustUpdate(t, tr, frameAt(0, det(100, 100, 60, 60, 0.9)))
	res := mustUpdate(t, tr, frameAt(1))
	assert.Empty(t, res.Exited)
	assert.Equal(t, 1, res.Tracks[0].Missed)

	// Two more misses: the third exceeds the 2-frame grace, the ID exits on the
	// very frame the limit is crossed.
	mustUpdate(t, tr, frameAt(2))
	res = mustUpdate(t, tr, frameAt(3))
	assert.Empty(t, res.Tracks)
	assert.Equal(t, []int{1}, res.Exited)

	// A face re-entering in the same place after the exit starts a new ID.
	res = mustUpdate(t, tr, frameAt(4, det(100, 100, 60, 60, 0.9)))
	assert.Equal(t, []int{2}, res.NewTracks)
}

func TestTracker_MaxMissedDurationBoundsCoasting(t *testing.T) {
	tr := newTestTracker(t, func(c *pigo.TrackerConfig) {
		c.MaxMissedFrames = pigo.MaxTrackerCandidates // frame-count grace is effectively off
		c.MaxMissedDuration = 500 * time.Millisecond
	})

	mustUpdate(t, tr, frameAt(0, det(100, 100, 60, 60, 0.9)))
	// 15 frames = 500ms at 30fps: still within the grace.
	res := mustUpdate(t, tr, frameAt(15))
	require.Len(t, res.Tracks, 1)
	assert.Equal(t, 1, res.Tracks[0].ID)

	// 17 frames = 566ms: the wall-clock grace is exceeded before matching, even
	// though the frame-count limit would have kept it.
	res = mustUpdate(t, tr, frameAt(17, det(100, 100, 60, 60, 0.9)))
	assert.Equal(t, []int{1}, res.Exited)
	assert.Equal(t, []int{2}, res.NewTracks, "a detection after the grace is a new identity")
	require.Len(t, res.Tracks, 1, "only the new track remains")
	assert.Equal(t, 2, res.Tracks[0].ID)
}

func TestTracker_ResolutionSwitchKeepsIDs(t *testing.T) {
	tr := newTestTracker(t, nil)

	const frames = 8
	for i := 0; i < frames; i++ {
		f := frameAt(i, det(100+2*float64(i), 80, 40, 40, 0.9), det(300, 80+3*float64(i), 40, 40, 0.9))
		f.Width, f.Height = 640, 480
		res := mustUpdate(t, tr, f)
		require.Equal(t, []int{1, 2}, res.Assignments, "frame %d", i)
	}

	// Upscale 2x. Detections arrive in the new coordinate system, so the
	// tracker must rescale its state instead of losing both faces.
	for i := frames; i < 2*frames; i++ {
		f := frameAt(i, det(200+4*float64(i-frames), 160, 80, 80, 0.9), det(600, 160+6*float64(i-frames), 80, 80, 0.9))
		f.Width, f.Height = 1280, 960
		res := mustUpdate(t, tr, f)
		require.Equal(t, []int{1, 2}, res.Assignments, "frame %d", i)
		assert.Empty(t, res.NewTracks)
	}

	// Switching back down must work just as well.
	for i := 2 * frames; i < 3*frames; i++ {
		f := frameAt(i, det(100+2*float64(i), 80, 40, 40, 0.9), det(300, 80+3*float64(i), 40, 40, 0.9))
		f.Width, f.Height = 640, 480
		res := mustUpdate(t, tr, f)
		require.Equal(t, []int{1, 2}, res.Assignments, "frame %d", i)
	}

	// Zero dimensions keep the last known resolution instead of resetting the
	// scaling, so a frame that omits size still matches.
	res := mustUpdate(t, tr, frameAt(3*frames, det(100+2*float64(3*frames), 80, 40, 40, 0.9)))
	assert.Equal(t, []int{1}, res.Assignments)
}

func TestTracker_ZoomKeepsID(t *testing.T) {
	tr := newTestTracker(t, nil)

	// A face approaching the camera: the rectangle grows by 8% per frame while
	// its center stays put. Size-aware cost plus a size velocity keeps the ID.
	for i := 0; i < 30; i++ {
		size := 40 * math.Pow(1.08, float64(i))
		cx, cy := 320.0, 240.0
		res := mustUpdate(t, tr, frameAt(i, det(cx-size/2, cy-size/2, size, size, 0.9)))
		require.Len(t, res.Tracks, 1, "frame %d", i)
		assert.Equal(t, []int{1}, res.Assignments, "frame %d", i)
	}

	// Hold still afterwards: no new identity despite the very different size
	// from the first frame.
	res := mustUpdate(t, tr, frameAt(30, det(320-200, 240-200, 400, 400, 0.9)))
	assert.Equal(t, []int{1}, res.Assignments)
}

func TestTracker_NewFaceDoesNotStealID(t *testing.T) {
	tr := newTestTracker(t, nil)

	for i := 0; i < 5; i++ {
		mustUpdate(t, tr, frameAt(i, det(100+5*float64(i), 100, 60, 60, 0.9)))
	}
	// A second face enters far away. The existing face must keep ID 1 and the
	// newcomer gets ID 2 even though the detections are ordered newcomer-first.
	for i := 5; i < 12; i++ {
		res := mustUpdate(t, tr, frameAt(i,
			det(700, 400, 60, 60, 0.95), // newcomer, higher confidence, listed first
			det(100+5*float64(i), 100, 60, 60, 0.9),
		))
		require.Equal(t, []int{1, 2}, trackIDs(res), "frame %d", i)
		assert.Equal(t, 2, res.Assignments[0], "newcomer keeps its own ID on frame %d", i)
		assert.Equal(t, 1, res.Assignments[1], "tracked face keeps ID 1 on frame %d", i)
	}
}

func TestTracker_DeterministicAcrossReplays(t *testing.T) {
	run := func() []pigo.FrameResult {
		tr := newTestTracker(t, nil)
		out := make([]pigo.FrameResult, 0, 10)
		for i := 0; i < 10; i++ {
			// Identical geometry for both faces in the first frame: ties must
			// resolve the same way every run.
			x := 100 + 10*float64(i)
			out = append(out, mustUpdate(t, tr, frameAt(i,
				det(600, 100, 60, 60, 0.7),
				det(x, 100, 60, 60, 0.7),
			)))
		}
		return out
	}
	first := run()
	for replay := 0; replay < 3; replay++ {
		assert.Equal(t, first, run(), "replay %d", replay)
	}
}

func TestTracker_InputValidationLeavesStateUntouched(t *testing.T) {
	tr := newTestTracker(t, func(c *pigo.TrackerConfig) { c.MaxDetections = 4; c.MaxTracks = 2 })

	res := mustUpdate(t, tr, frameAt(0, det(100, 100, 60, 60, 0.9)))
	require.Equal(t, []int{1}, res.NewTracks)

	// Every rejected frame must leave the tracker exactly as it was: the same
	// clock, the same tracks and the same next ID.
	bad := []struct {
		name string
		f    pigo.Frame
		err  error
	}{
		{
			"time regression",
			frameAt(-1, det(100, 100, 60, 60, 0.9)),
			pigo.ErrTimeRegression,
		},
		{
			"duplicate frame",
			frameAt(0, det(100, 100, 60, 60, 0.9)),
			pigo.ErrDuplicateTimestamp,
		},
		{
			"invalid rect zero width",
			frameAt(1, det(100, 100, 0, 60, 0.9)),
			pigo.ErrInvalidDetection,
		},
		{
			"invalid rect negative",
			frameAt(1, det(100, 100, -5, 60, 0.9)),
			pigo.ErrInvalidDetection,
		},
		{
			"invalid rect NaN",
			frameAt(1, det(math.NaN(), 100, 60, 60, 0.9)),
			pigo.ErrInvalidDetection,
		},
		{
			"invalid rect Inf",
			frameAt(1, det(math.Inf(1), 100, 60, 60, 0.9)),
			pigo.ErrInvalidDetection,
		},
		{
			"confidence above one",
			frameAt(1, det(100, 100, 60, 60, 1.5)),
			pigo.ErrInvalidDetection,
		},
		{
			"confidence NaN",
			frameAt(1, det(100, 100, 60, 60, math.NaN())),
			pigo.ErrInvalidDetection,
		},
		{
			"half frame size",
			pigo.Frame{Timestamp: trackBase.Add(time.Second / 30), Width: 640},
			pigo.ErrInvalidFrameSize,
		},
		{
			"negative frame size",
			pigo.Frame{Timestamp: trackBase.Add(time.Second / 30), Width: -1, Height: -1},
			pigo.ErrInvalidFrameSize,
		},
		{
			"too many detections",
			frameAt(1, det(1, 1, 10, 10, 0.9), det(20, 1, 10, 10, 0.9),
				det(40, 1, 10, 10, 0.9), det(60, 1, 10, 10, 0.9), det(80, 1, 10, 10, 0.9)),
			pigo.ErrTooManyDetections,
		},
	}
	for _, tc := range bad {
		_, err := tr.Update(tc.f)
		require.ErrorIs(t, err, tc.err, tc.name)
	}

	// The original detection, at the original timestamp, is still accepted and
	// keeps the original ID; the rejected frames neither advanced the clock nor
	// consumed track slots.
	res = mustUpdate(t, tr, frameAt(1, det(104, 100, 60, 60, 0.9)))
	assert.Equal(t, []int{1}, res.Assignments)
	assert.Empty(t, res.NewTracks)
	assert.Empty(t, res.Exited)
}

func TestTracker_TrackLimitIsAtomic(t *testing.T) {
	tr := newTestTracker(t, func(c *pigo.TrackerConfig) { c.MaxTracks = 2 })

	mustUpdate(t, tr, frameAt(0, det(10, 10, 20, 20, 0.9), det(100, 10, 20, 20, 0.9)))

	_, err := tr.Update(frameAt(1,
		det(12, 10, 20, 20, 0.9),
		det(102, 10, 20, 20, 0.9),
		det(300, 300, 20, 20, 0.9),
	))
	assert.ErrorIs(t, err, pigo.ErrTrackLimitExceeded)

	// Nothing was consumed: the same frame without the newcomer still matches
	// both original IDs.
	res := mustUpdate(t, tr, frameAt(1, det(12, 10, 20, 20, 0.9), det(102, 10, 20, 20, 0.9)))
	assert.Equal(t, []int{1, 2}, res.Assignments)
}

func TestTracker_ExitedSlotIsReusableInTheSameFrame(t *testing.T) {
	tr := newTestTracker(t, func(c *pigo.TrackerConfig) { c.MaxTracks = 1; c.MaxMissedFrames = 0 })

	mustUpdate(t, tr, frameAt(0, det(10, 10, 20, 20, 0.9)))
	// The previous track exceeds its zero-frame grace and a new face is already
	// present: the slot is freed in the same frame instead of failing.
	res := mustUpdate(t, tr, frameAt(1, det(400, 400, 20, 20, 0.9)))
	assert.Equal(t, []int{1}, res.Exited)
	assert.Equal(t, []int{2}, res.NewTracks)
	assert.Equal(t, []int{2}, res.Assignments)
}

func TestTracker_MaxCandidatesIsBounded(t *testing.T) {
	tr := newTestTracker(t, func(c *pigo.TrackerConfig) {
		c.MaxDetections = pigo.MaxTrackerCandidates
		c.MaxTracks = pigo.MaxTrackerCandidates
	})

	dets := make([]pigo.TrackDetection, 0, pigo.MaxTrackerCandidates)
	for i := 0; i < pigo.MaxTrackerCandidates; i++ {
		dets = append(dets, det(float64(i%16)*20, float64(i/16)*20, 12, 12, 0.5))
	}
	res := mustUpdate(t, tr, frameAt(0, dets...))
	assert.Len(t, res.Tracks, pigo.MaxTrackerCandidates)

	// The per-frame cap is checked before the quadratic allocation, so oversized
	// input stays cheap to reject.
	over := append(dets, det(500, 500, 12, 12, 0.5))
	_, err := tr.Update(frameAt(1, over...))
	assert.ErrorIs(t, err, pigo.ErrTooManyDetections)
}

func TestTracker_Reset(t *testing.T) {
	tr := newTestTracker(t, nil)
	mustUpdate(t, tr, frameAt(0, det(100, 100, 60, 60, 0.9)))

	tr.Reset()
	res := mustUpdate(t, tr, frameAt(0, det(100, 100, 60, 60, 0.9)))
	assert.Equal(t, []int{1}, res.NewTracks, "IDs restart after Reset")
	assert.Equal(t, 1, res.Tracks[0].Age)
}

func TestTracker_PredictionHorizonFreezesCoastingRect(t *testing.T) {
	tr := newTestTracker(t, func(c *pigo.TrackerConfig) {
		c.MaxMissedFrames = 60
		c.PredictionHorizon = 100 * time.Millisecond
	})

	mustUpdate(t, tr, frameAt(0, det(0, 0, 20, 20, 0.9)))
	mustUpdate(t, tr, frameAt(1, det(8, 0, 20, 20, 0.9))) // 240 px/s to the right

	// Within the horizon the prediction moves.
	res := mustUpdate(t, tr, frameAt(2))
	require.Len(t, res.Tracks, 1)
	assert.InDelta(t, 16, res.Tracks[0].Rect.X, 1e-6)

	// Once the horizon (100ms = 24 px) is exhausted, further frames report the
	// same frozen prediction instead of extrapolating without bound.
	res = mustUpdate(t, tr, frameAt(10)) // 300ms since last sighting
	require.Len(t, res.Tracks, 1)
	frozen := res.Tracks[0].Rect.X
	assert.InDelta(t, 32, frozen, 1e-6)
	res = mustUpdate(t, tr, frameAt(30))
	require.Len(t, res.Tracks, 1)
	assert.InDelta(t, frozen, res.Tracks[0].Rect.X, 1e-9)
	assert.True(t, res.Tracks[0].Rect.Valid())
}

func TestTracker_DetectionRectFromCascade(t *testing.T) {
	// pigo detections carry the window center in Row/Col and the side length in
	// Scale; the adapter must produce the matching top-left rectangle.
	d := pigo.Detection{Row: 100, Col: 80, Scale: 40, Q: 12.5}
	assert.Equal(t, pigo.Rect{X: 60, Y: 80, Width: 40, Height: 40}, pigo.DetectionRect(d))

	assert.False(t, pigo.DetectionRect(pigo.Detection{Scale: 0}).Valid())
}

func TestTracker_ConcurrentTrackersAreIndependent(t *testing.T) {
	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(offset float64) {
			defer wg.Done()
			tr, err := pigo.NewTracker(pigo.DefaultTrackerConfig())
			if !assert.NoError(t, err) {
				return
			}
			for i := 0; i < 30; i++ {
				res, err := tr.Update(frameAt(i, det(offset+float64(i), 50, 30, 30, 0.8)))
				if !assert.NoError(t, err) {
					return
				}
				assert.Equal(t, []int{1}, res.Assignments)
			}
		}(float64(100 * g))
	}
	wg.Wait()
}

// TestTracker_CascadeSequence feeds real cascade detections through the tracker
// and checks that the identity survives the original detection pipeline change
// in resolution and the two-face crossing scenario. It also guards the existing
// cascade tests by running them through the same package-level fixtures.
func TestTracker_CascadeSequence(t *testing.T) {
	p := pigo.NewPigo().Unpack()

	// A base frame with a real face, at the sample image resolution.
	cParams := *cParams
	base := p.RunCascade(cParams, 0.0)
	require.NotEmpty(t, p.ClusterDetections(base, 0.1))

	dets := p.ClusterDetections(p.RunCascade(cParams, 0.0), 0.1)
	require.NotEmpty(t, dets)
	face := pigo.DetectionRect(dets[0])
	require.True(t, face.Valid())

	tr := newTestTracker(t, func(c *pigo.TrackerConfig) {
		c.MaxMissedFrames = 3
		c.MatchThreshold = 1.5
	})

	// Frame 0 at the native resolution.
	f := pigo.Frame{
		Timestamp:  trackBase.Add(0),
		Width:      cParams.Cols,
		Height:     cParams.Rows,
		Detections: []pigo.TrackDetection{{Rect: face, Confidence: 1}},
	}
	res := mustUpdate(t, tr, f)
	require.Len(t, res.Tracks, 1)
	id := res.Tracks[0].ID

	// Frame 1: same face after a 2x resolution switch. The tracker must rescale,
	// not create a new ID.
	scaled := pigo.Rect{X: face.X * 2, Y: face.Y * 2, Width: face.Width * 2, Height: face.Height * 2}
	res = mustUpdate(t, tr, pigo.Frame{
		Timestamp:  trackBase.Add(time.Second / 30),
		Width:      cParams.Cols * 2,
		Height:     cParams.Rows * 2,
		Detections: []pigo.TrackDetection{{Rect: scaled, Confidence: 0.9}},
	})
	assert.Equal(t, []int{id}, res.Assignments)

	// Frame 2: the face is detected again while a second, far-away face enters.
	// The newcomer must get its own identity without disturbing the tracked one.
	other := pigo.Rect{
		X:      face.X*2 + float64(cParams.Cols),
		Y:      face.Y * 2,
		Width:  face.Width * 2,
		Height: face.Height * 2,
	}
	res = mustUpdate(t, tr, pigo.Frame{
		Timestamp: trackBase.Add(2 * time.Second / 30),
		Width:     cParams.Cols * 2,
		Height:    cParams.Rows * 2,
		Detections: []pigo.TrackDetection{
			{Rect: scaled, Confidence: 0.9},
			{Rect: other, Confidence: 0.8},
		},
	})
	require.Len(t, res.Tracks, 2)
	assert.Equal(t, id, res.Assignments[0])
	assert.NotEqual(t, id, res.Assignments[1])

	// Frame 3-5: the original face is occluded for three frames but coasts.
	for i := 3; i <= 5; i++ {
		res = mustUpdate(t, tr, pigo.Frame{
			Timestamp: trackBase.Add(time.Duration(i) * time.Second / 30),
			Detections: []pigo.TrackDetection{
				{Rect: other, Confidence: 0.8},
			},
		})
		tk, ok := trackByID(res, id)
		require.True(t, ok, "frame %d", i)
		assert.True(t, tk.Coasting)
		assert.Equal(t, 1, tk.ID)
	}

	// Frame 6: occluder gone, face back: same ID, no new track.
	res = mustUpdate(t, tr, pigo.Frame{
		Timestamp: trackBase.Add(6 * time.Second / 30),
		Detections: []pigo.TrackDetection{
			{Rect: scaled, Confidence: 0.9},
			{Rect: other, Confidence: 0.8},
		},
	})
	assert.Equal(t, []int{id, id + 1}, res.Assignments)
}

func ExampleTracker() {
	cfg := pigo.DefaultTrackerConfig()
	cfg.MatchThreshold = 1.2
	cfg.MaxMissedFrames = 5
	tr, err := pigo.NewTracker(cfg)
	if err != nil {
		panic(err)
	}

	base := time.Date(2026, 11, 20, 12, 0, 0, 0, time.UTC)
	for i := 0; i < 3; i++ {
		res, err := tr.Update(pigo.Frame{
			Timestamp:  base.Add(time.Duration(i) * 33 * time.Millisecond),
			Width:      1280,
			Height:     720,
			Detections: []pigo.TrackDetection{{Rect: pigo.Rect{X: 100 + 4*float64(i), Y: 200, Width: 80, Height: 80}, Confidence: 0.95}},
		})
		if err != nil {
			panic(err)
		}
		tk := res.Tracks[0]
		fmt.Printf("frame %d: id=%d x=%.0f\n", i, tk.ID, tk.Rect.X)
	}
	// Output:
	// frame 0: id=1 x=100
	// frame 1: id=1 x=104
	// frame 2: id=1 x=108
}

func TestTracker_ErrorMessagesAreDistinct(t *testing.T) {
	// Errors are part of the public contract; make sure none of them collapse
	// into another through errors.Is.
	all := []error{
		pigo.ErrTimeRegression,
		pigo.ErrDuplicateTimestamp,
		pigo.ErrInvalidFrameSize,
		pigo.ErrInvalidDetection,
		pigo.ErrTooManyDetections,
		pigo.ErrTrackLimitExceeded,
		pigo.ErrInvalidTrackerConfig,
	}
	for i := range all {
		for j := range all {
			if i == j {
				continue
			}
			assert.False(t, errors.Is(all[i], all[j]), "%v vs %v", all[i], all[j])
		}
	}
}
