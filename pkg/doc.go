/*
Package pigo is a lightweight pure Go face detection, pupil/eyes localization and facial landmark points detection library
based on Pixel Intensity Comparison-based Object detection paper (https://arxiv.org/pdf/1305.4537.pdf).
Is platform agnostic and does not require any external dependencies and third party modules.

Face detection API example.

First the source image should be converted to grayscale mode, then the cascade function can be invoked, which returns a slice containing the row, column, the scale and the detection score.

	src, err := pigo.GetImage("/path/to/image")
	if err != nil {
		log.Fatalf("Cannot open the image file: %v", err)
	}

	pixels := pigo.RgbToGrayscale(src)
	cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

	cParams := pigo.CascadeParams{
		MinSize:     20,
		MaxSize:     1000,
		ShiftFactor: 0.1,
		ScaleFactor: 1.1,

		ImageParams: pigo.ImageParams{
			Pixels: pixels,
			Rows:   rows,
			Cols:   cols,
			Dim:    cols,
		},
	}

	p := pigo.NewPigo()
	// Unpack the binary file. This will return the number of cascade trees,
	// the tree depth, the threshold and the prediction from tree's leaf nodes.
	classifier := p.Unpack()

	angle := 0.0 // cascade rotation angle. 0.0 is 0 radians and 1.0 is 2*pi radians

	// Run the classifier over the obtained leaf nodes and return the detection results.
	// The result contains quadruplets representing the row, column, scale and detection score.
	dets := classifier.RunCascade(cParams, angle)

	// Calculate the intersection over union (IoU) of two clusters.
	dets = classifier.ClusterDetections(dets, 0.2)

Face tracking API example.

RunCascade and ClusterDetections answer "is there a face in this frame", not
"is it the same face as in the previous frame". Tracker adds that: it consumes
the clustered detections of each frame (converted with DetectionRect) plus a
timestamp, and returns stable track IDs, the current or predicted rectangle and
a smoothed confidence. Crossing faces, zooming faces and new faces are matched
globally against predicted track positions, and a track survives a configurable
number of missed frames.

	// Cascade Q scores are unbounded; normalize them to [0, 1] for the tracker.
	maxQ := float32(0)
	for _, d := range dets {
		maxQ = max(maxQ, d.Q)
	}
	frameDets := make([]TrackDetection, len(dets))
	for i, d := range dets {
		frameDets[i] = TrackDetection{
			Rect:       pigo.DetectionRect(d),
			Confidence: float64(d.Q / maxQ),
		}
	}

	tracker, err := pigo.NewTracker(pigo.DefaultTrackerConfig())
	if err != nil {
		log.Fatalf("Cannot create tracker: %v", err)
	}

	res, err := tracker.Update(pigo.Frame{
		Timestamp:  time.Now(),
		Width:      cols,
		Height:     rows,
		Detections: frameDets,
	})
	if err != nil {
		log.Fatalf("Cannot update tracker: %v", err)
	}

	for _, track := range res.Tracks {
		log.Printf("id=%d rect=%+v confidence=%.2f coasting=%v", track.ID, track.Rect, track.Confidence, track.Coasting)
	}

For pupil/eyes localization and facial landmark points detection API example check the source code.
*/
package pigo
