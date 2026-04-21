from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from track.world_anchor import AnchorState


class HomographyAnchor:
    def __init__(
        self,
        min_inliers: int = 10,
        min_inlier_ratio: float = 0.3,
        lost_frames_threshold: int = 15,
        margin_ratio: float = 0.25,
        orb_nfeatures: int = 500,
        ratio_test: float = 0.75,
        ransac_reproj_threshold: float = 5.0,
        retry_interval: int = 5,
    ) -> None:
        self._min_inliers = min_inliers
        self._min_inlier_ratio = min_inlier_ratio
        self._lost_frames_threshold = lost_frames_threshold
        self._margin_ratio = margin_ratio
        self._ratio_test = ratio_test
        self._ransac_threshold = ransac_reproj_threshold
        self._retry_interval = retry_interval

        self._orb = cv2.ORB_create(nfeatures=orb_nfeatures)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self._ref_keypoints: Optional[list] = None
        self._ref_descriptors: Optional[np.ndarray] = None
        self._last_homography: Optional[np.ndarray] = None
        self._lost_frames = 0
        self._frames_since_retry = 0
        self._lost = True  # remains True until successful initialization

    def initialize(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> None:
        x, y, w, h = region
        mx = int(w * self._margin_ratio)
        my = int(h * self._margin_ratio)
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(frame.shape[1], x + w + mx)
        y1 = min(frame.shape[0], y + h + my)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        kp, des = self._orb.detectAndCompute(gray, mask)

        if des is None or len(kp) < self._min_inliers:
            self._lost = True
            return

        self._ref_keypoints = kp
        self._ref_descriptors = des
        self._last_homography = np.eye(3, dtype=np.float64)
        self._lost_frames = 0
        self._lost = False

    def update(self, frame: np.ndarray) -> AnchorState:
        if self._ref_descriptors is None:
            return AnchorState(homography=None, confidence=0.0, lost=True)

        if self._lost:
            self._frames_since_retry += 1
            if self._frames_since_retry < self._retry_interval:
                return AnchorState(homography=None, confidence=0.0, lost=True)
            self._frames_since_retry = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kp) < self._min_inliers:
            return self._handle_fail()

        matches = self._matcher.knnMatch(self._ref_descriptors, des, k=2)
        good = [
            m for m_pair in matches
            if len(m_pair) == 2
            for m, n in [m_pair]
            if m.distance < self._ratio_test * n.distance
        ]

        if len(good) < self._min_inliers:
            return self._handle_fail()

        src_pts = np.float32(
            [self._ref_keypoints[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, inlier_mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self._ransac_threshold
        )
        if H is None or inlier_mask is None:
            return self._handle_fail()

        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / len(good)
        if inliers < self._min_inliers or inlier_ratio < self._min_inlier_ratio:
            return self._handle_fail()

        self._last_homography = H
        self._lost_frames = 0
        self._lost = False  # re-acquisition path: revive from lost state
        self._frames_since_retry = 0
        return AnchorState(homography=H, confidence=inlier_ratio, lost=False)

    def _handle_fail(self) -> AnchorState:
        if self._lost:
            return AnchorState(homography=None, confidence=0.0, lost=True)
        self._lost_frames += 1
        if self._lost_frames >= self._lost_frames_threshold:
            self._lost = True
            self._frames_since_retry = 0
            return AnchorState(homography=None, confidence=0.0, lost=True)
        return AnchorState(homography=self._last_homography, confidence=0.0, lost=False)

    def is_lost(self) -> bool:
        return self._lost
