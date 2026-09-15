"""
Ecological Event Sampling Strategy for action recognition
Implements multiple sampling strategies for video frame selection
"""

import numpy as np
from PIL import Image


class EcologicalEventSampler:
    """
    Sampler that implements multiple strategies for selecting frames from video:
    - uniform: evenly spaced frames (baseline)
    - random: randomly selected frames
    - motion: frames with high motion activity
    - relation: frames with high inter-frame relationships
    - ees: full Ecological Event Sampling (motion + novelty + relation + uncertainty)
    """
    
    def __init__(
        self,
        strategy: str = "uniform",
        scout_max_frames: int = 64,
        event_window_radius: int = 1,
    ):
        """
        Args:
            strategy: sampling strategy name
            scout_max_frames: max frames for preview stage in EES
            event_window_radius: temporal radius around event peaks
        """
        self.strategy = strategy
        self.scout_max_frames = scout_max_frames
        self.event_window_radius = event_window_radius
        
        assert strategy in ["uniform", "random", "motion", "relation", "ees"], \
            f"Unknown strategy: {strategy}"
    
    def sample_indices(self, num_frames: int, budget_k: int, frame_paths=None) -> np.ndarray:
        """
        Sample frame indices from video
        
        Args:
            num_frames: total number of frames in video
            budget_k: target number of frames to select
            frame_paths: list of frame file paths (optional, for EES)
        
        Returns:
            array of selected frame indices, sorted
        """
        if num_frames <= 0:
            raise ValueError("A video must contain at least one frame.")
        if self.strategy == "uniform":
            indices = self._uniform_indices(num_frames, budget_k)
        elif self.strategy == "random":
            indices = self._random_indices(num_frames, budget_k)
        elif self.strategy == "motion":
            if frame_paths is None:
                indices = self._motion_indices_heuristic(num_frames, budget_k)
            else:
                indices = self._motion_indices(num_frames, budget_k, frame_paths)
        elif self.strategy == "relation":
            if frame_paths is None:
                indices = self._relation_indices_heuristic(num_frames, budget_k)
            else:
                indices = self._relation_indices(num_frames, budget_k, frame_paths)
        elif self.strategy == "ees":
            if frame_paths is None:
                indices = self._uniform_indices(num_frames, budget_k)
            else:
                indices = self._ees_indices(num_frames, budget_k, frame_paths)
        else:
            indices = self._uniform_indices(num_frames, budget_k)
        # The transform expects exactly K frames. Repeating an index is necessary
        # only for clips shorter than the requested temporal budget.
        return np.resize(np.sort(indices).astype(int), budget_k)
    
    def _uniform_indices(self, num_frames: int, budget_k: int) -> np.ndarray:
        """Baseline: evenly spaced frames"""
        if num_frames <= budget_k:
            # If video shorter than budget, repeat interpolation
            indices = np.linspace(0, num_frames - 1, budget_k, dtype=int)
        else:
            # Standard linspace
            indices = np.linspace(0, num_frames - 1, budget_k, dtype=int)
        return np.unique(np.sort(indices))
    
    def _random_indices(self, num_frames: int, budget_k: int) -> np.ndarray:
        """Random frame selection"""
        indices = np.random.choice(num_frames, size=min(budget_k, num_frames), replace=False)
        return np.sort(indices)
    
    def _motion_indices_heuristic(self, num_frames: int, budget_k: int) -> np.ndarray:
        """Motion-based sampling (heuristic, no images)"""
        # Without images, approximate motion distribution
        # High motion typically at transitions: sample more densely around boundaries
        indices = []
        
        # Add anchor frames
        if num_frames >= 2:
            indices.append(0)
            indices.append(num_frames - 1)
        
        # Fill rest with motion-like distribution (favor transitions)
        remaining = budget_k - len(indices)
        if remaining > 0:
            # Create synthetic motion signal (emphasize transitions)
            motion_signal = np.abs(np.diff(np.linspace(0, 1, num_frames)))
            motion_signal = np.concatenate([[motion_signal[0]], motion_signal])
            
            # Top-k frames by motion
            top_k = np.argsort(motion_signal)[-remaining:]
            indices.extend(top_k.tolist())
        
        return np.sort(np.unique(np.array(indices))).astype(int)
    
    def _motion_indices(
        self,
        num_frames: int,
        budget_k: int,
        frame_paths: list
    ) -> np.ndarray:
        """Motion-based sampling using frame differences"""
        # Load scout frames
        scout_indices = self._scout_preview_indices(num_frames)
        scout_images = self._load_scout_images(frame_paths, scout_indices)
        
        if scout_images is None:
            return self._uniform_indices(num_frames, budget_k)
        
        # Compute motion scores
        motion_scores = self._compute_motion_scores(scout_images)
        
        # Select peaks
        selected_indices = self._select_event_peaks(
            motion_scores, budget_k, scout_indices
        )
        
        return np.sort(selected_indices).astype(int)
    
    def _relation_indices_heuristic(self, num_frames: int, budget_k: int) -> np.ndarray:
        """Relation-based sampling (heuristic, no images)"""
        # Without images, use temporal distance as proxy for relationship diversity
        indices = []
        
        # Stratified sampling: divide into regions and sample from each
        region_size = max(1, num_frames // budget_k)
        for i in range(budget_k):
            start = i * region_size
            end = min((i + 1) * region_size, num_frames)
            if start < num_frames:
                idx = np.random.randint(start, end)
                indices.append(idx)
        
        return np.sort(np.unique(np.array(indices))).astype(int)
    
    def _relation_indices(
        self,
        num_frames: int,
        budget_k: int,
        frame_paths: list
    ) -> np.ndarray:
        """Relation-based sampling using scene diversity"""
        scout_indices = self._scout_preview_indices(num_frames)
        scout_images = self._load_scout_images(frame_paths, scout_indices)
        
        if scout_images is None:
            return self._uniform_indices(num_frames, budget_k)
        
        relation_scores = self._compute_relation_scores(scout_images)
        
        selected_indices = self._select_event_peaks(
            relation_scores, budget_k, scout_indices
        )
        
        return np.sort(selected_indices).astype(int)
    
    def _ees_indices(
        self,
        num_frames: int,
        budget_k: int,
        frame_paths: list
    ) -> np.ndarray:
        """Full EES: motion + novelty + relation + uncertainty"""
        # Scout preview
        scout_indices = self._scout_preview_indices(num_frames)
        scout_images = self._load_scout_images(frame_paths, scout_indices)
        
        if scout_images is None:
            return self._uniform_indices(num_frames, budget_k)
        
        # Compute all scores
        motion_scores = self._compute_motion_scores(scout_images)
        scout_features = self._compute_scout_features(scout_images)
        novelty_scores = self._compute_novelty_scores(scout_features)
        relation_scores = self._compute_relation_scores(scout_images)
        uncertainty_scores = self._compute_uncertainty_scores(scout_images)
        
        # Eq. (4): motion + novelty + relation + 0.5 * uncertainty.
        combined_scores = (
            motion_scores +
            novelty_scores +
            relation_scores +
            0.5 * uncertainty_scores
        )
        
        # Normalize
        combined_scores = (combined_scores - combined_scores.min()) / \
                         (combined_scores.max() - combined_scores.min() + 1e-8)
        
        # Retain four event peaks and their radius-one temporal neighborhoods.
        peak_count = min(4, len(scout_indices))
        peak_positions = np.argsort(combined_scores)[-peak_count:]
        selected_indices = self._expand_with_window(scout_indices[peak_positions], num_frames)
        
        # Add anchor frames
        selected_indices = self._add_anchor_frames(
            selected_indices, num_frames, budget_k
        )
        
        # Preserve event evidence when candidates exceed the budget; fill remaining
        # slots uniformly to maintain global temporal coverage.
        selected_indices = np.unique(selected_indices)
        if len(selected_indices) > budget_k:
            scores_by_index = {int(index): score for index, score in zip(scout_indices, combined_scores)}
            selected_indices = np.array(sorted(selected_indices, key=lambda index: scores_by_index.get(int(index), 0.0), reverse=True)[:budget_k])
        if len(selected_indices) < budget_k:
            selected = list(selected_indices)
            for candidate in np.linspace(0, num_frames - 1, budget_k, dtype=int):
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) == budget_k:
                    break
            selected_indices = np.array(selected)
        return np.sort(selected_indices).astype(int)
    
    def _scout_preview_indices(self, num_frames: int) -> np.ndarray:
        """Get indices for preview stage"""
        if num_frames <= self.scout_max_frames:
            return np.arange(num_frames, dtype=int)
        else:
            return np.linspace(0, num_frames - 1, self.scout_max_frames, dtype=int)
    
    def _load_scout_images(self, frame_paths: list, scout_indices: np.ndarray):
        """Load scout images for analysis"""
        try:
            images = []
            for idx in scout_indices:
                if idx < len(frame_paths):
                    img = Image.open(frame_paths[idx]).convert('RGB')
                    # Resize for efficiency
                    img = img.resize((56, 56), Image.Resampling.BILINEAR)
                    images.append(np.array(img, dtype=np.float32) / 255.0)
            return np.array(images) if images else None
        except Exception as e:
            print(f"[WARNING] Failed to load scout images: {e}")
            return None
    
    def _compute_motion_scores(self, images: np.ndarray) -> np.ndarray:
        """Compute motion scores based on frame differences"""
        if len(images) < 2:
            return np.ones(len(images))
        
        # Convert to grayscale
        gray = np.mean(images, axis=3)  # [N, H, W]
        
        # Frame differences
        diffs = np.abs(np.diff(gray, axis=0))  # [N-1, H, W]
        
        # Motion score per frame (mean absolute difference)
        difference_scores = np.mean(diffs, axis=(1, 2))
        motion_scores = np.empty(len(images), dtype=np.float32)
        motion_scores[0] = difference_scores[0]
        motion_scores[-1] = difference_scores[-1]
        if len(images) > 2:
            motion_scores[1:-1] = (difference_scores[:-1] + difference_scores[1:]) / 2
        
        # Normalize
        motion_scores = (motion_scores - motion_scores.min()) / \
                       (motion_scores.max() - motion_scores.min() + 1e-8)
        
        return motion_scores
    
    def _compute_scout_features(self, images: np.ndarray) -> np.ndarray:
        """Extract simple features from scout images"""
        features = []
        for img in images:
            # Simple histogram-based feature
            hist_r = np.histogram(img[:, :, 0], bins=8)[0]
            hist_g = np.histogram(img[:, :, 1], bins=8)[0]
            hist_b = np.histogram(img[:, :, 2], bins=8)[0]
            feat = np.concatenate([hist_r, hist_g, hist_b])
            feat = feat / (np.sum(feat) + 1e-8)
            features.append(feat)
        return np.array(features)
    
    def _compute_novelty_scores(self, features: np.ndarray) -> np.ndarray:
        """Compute novelty scores (diversity from neighbors)"""
        scores = np.zeros(len(features))
        
        if len(features) < 2:
            return scores
        
        for i in range(len(features)):
            # Compare with neighbors
            if i == 0:
                dist = np.linalg.norm(features[i] - features[i + 1])
            elif i == len(features) - 1:
                dist = np.linalg.norm(features[i] - features[i - 1])
            else:
                dist1 = np.linalg.norm(features[i] - features[i - 1])
                dist2 = np.linalg.norm(features[i] - features[i + 1])
                dist = (dist1 + dist2) / 2
            
            scores[i] = dist
        
        # Normalize
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores
    
    def _compute_relation_scores(self, images: np.ndarray) -> np.ndarray:
        """Approximate interaction activation with moving-region count and spread."""
        if len(images) < 2:
            return np.zeros(len(images), dtype=np.float32)
        gray = images.mean(axis=3)
        scores = np.zeros(len(images), dtype=np.float32)
        for frame_id in range(1, len(images)):
            difference = np.abs(gray[frame_id] - gray[frame_id - 1])
            foreground = difference > (difference.mean() + difference.std())
            visited = np.zeros_like(foreground, dtype=bool)
            centroids, components = [], 0
            for y, x in zip(*np.where(foreground & ~visited)):
                if visited[y, x]:
                    continue
                components += 1
                stack, pixels = [(y, x)], []
                visited[y, x] = True
                while stack:
                    py, px = stack.pop()
                    pixels.append((py, px))
                    for ny, nx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
                        if 0 <= ny < foreground.shape[0] and 0 <= nx < foreground.shape[1] and foreground[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                centroids.append(np.mean(pixels, axis=0))
            dispersion = np.std(centroids, axis=0).mean() if len(centroids) > 1 else 0.0
            scores[frame_id] = components + foreground.mean() + dispersion / max(foreground.shape)
        scores[0] = scores[1]
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    def _compute_uncertainty_scores(self, images: np.ndarray) -> np.ndarray:
        """Estimate observation uncertainty from sharpness, brightness, and contrast."""
        gray = images.mean(axis=3)
        brightness = gray.mean(axis=(1, 2))
        contrast = gray.std(axis=(1, 2))
        laplacian = 4 * gray[:, 1:-1, 1:-1] - gray[:, :-2, 1:-1] - gray[:, 2:, 1:-1] - gray[:, 1:-1, :-2] - gray[:, 1:-1, 2:]
        sharpness = laplacian.var(axis=(1, 2))
        sharpness = (sharpness - sharpness.min()) / (sharpness.max() - sharpness.min() + 1e-8)
        contrast = (contrast - contrast.min()) / (contrast.max() - contrast.min() + 1e-8)
        uncertainty = (1 - sharpness) + (1 - contrast) + 2 * np.abs(brightness - .5)
        return (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
    
    def _select_event_peaks(
        self,
        scores: np.ndarray,
        budget_k: int,
        scout_indices: np.ndarray
    ) -> np.ndarray:
        """Select top-k frames by score"""
        if len(scores) <= budget_k:
            return scout_indices
        
        top_k_indices = np.argsort(scores)[-budget_k:]
        return scout_indices[top_k_indices]
    
    def _expand_with_window(
        self,
        peak_indices: np.ndarray,
        num_frames: int
    ) -> np.ndarray:
        """Expand peaks with temporal window"""
        expanded = set()
        for peak in peak_indices:
            for offset in range(-self.event_window_radius, self.event_window_radius + 1):
                idx = peak + offset
                if 0 <= idx < num_frames:
                    expanded.add(idx)
        return np.array(sorted(expanded), dtype=int)
    
    def _add_anchor_frames(
        self,
        selected_indices: np.ndarray,
        num_frames: int,
        budget_k: int
    ) -> np.ndarray:
        """Add the two global anchors specified for Eco-Mamba (1/3 and 2/3)."""
        anchors = set(selected_indices.tolist())
        
        if num_frames > 1:
            anchors.add(round((num_frames - 1) / 3))
            anchors.add(round(2 * (num_frames - 1) / 3))
        
        # Convert back to array and trim to budget
        result = np.array(sorted(anchors), dtype=int)
        return result[:budget_k]
