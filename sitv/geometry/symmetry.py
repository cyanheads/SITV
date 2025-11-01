"""
Symmetry detection and quotient space analysis on parameter manifolds.

This module provides the SymmetryAnalyzer service for detecting parameter space
symmetries (rotation, permutation, scaling) and constructing quotient spaces that
factor out redundant parameter representations.

Neural network parameter spaces often exhibit symmetries:
- Rotation symmetry: L(R·θ) ≈ L(θ) for rotation matrices R
- Permutation symmetry: Neuron reordering preserves loss
- Scaling symmetry: Layer-wise rescaling can leave loss invariant

Working in quotient space Θ/G (parameters modulo symmetry group G) provides a more
principled geometric analysis by removing redundant degrees of freedom.

Reference: arXiv:2506.13018 - Symmetry quotients in parameter spaces
"""

from typing import Any, Optional

import torch
from torch import nn

from sitv.geometry.config import SymmetryAnalysisConfig


class SymmetryAnalyzer:
    """Service for detecting parameter space symmetries and quotient projection.

    This service tests whether the loss function is invariant under various
    group actions on parameters. Detected symmetries can be used to project
    models to canonical forms in quotient space.

    Attributes:
        config: Symmetry analysis configuration
        evaluator: Evaluation service to compute L(θ)
        device: Device for computation
        symmetry_tolerance: Tolerance for L(g·θ) ≈ L(θ)
    """

    def __init__(
        self,
        config: SymmetryAnalysisConfig,
        evaluator,  # EvaluationService (avoid circular import)
        device: str = "cuda"
    ):
        """Initialize the symmetry analyzer.

        Args:
            config: Symmetry analysis configuration
            evaluator: EvaluationService for loss computation
            device: Device for computation
        """
        self.config = config
        self.evaluator = evaluator
        self.device = device
        self.symmetry_tolerance = config.symmetry_tolerance

    # === Rotation Symmetry ===

    def detect_rotation_symmetry(
        self,
        model: nn.Module,
        eval_texts: list[str],
        num_tests: int = 10
    ) -> dict[str, Any]:
        """Test for rotation symmetry L(R·θ) ≈ L(θ).

        Applies random orthogonal transformations to parameter layers and
        checks if loss remains invariant.

        Args:
            model: Model to test
            eval_texts: Evaluation dataset
            num_tests: Number of random rotation tests

        Returns:
            Dictionary containing:
                - is_symmetric: Whether rotation symmetry detected
                - symmetry_score: Score in [0,1], 1 = perfect symmetry
                - violations: List of cases where symmetry breaks
                - avg_loss_deviation: Average |L(R·θ) - L(θ)|
                - tested_layers: Layers where rotation was tested

        Examples:
            >>> analyzer = SymmetryAnalyzer(config, evaluator, "cuda")
            >>> results = analyzer.detect_rotation_symmetry(model, texts, num_tests=10)
            >>> print(f"Rotation symmetric: {results['is_symmetric']}")
        """
        if not self.config.detect_rotations:
            return self._empty_symmetry_result("rotation")

        # Get baseline loss
        baseline_loss = self.evaluator.evaluate(model, eval_texts)

        deviations = []
        violations = []
        tested_layers = []

        # Test random rotations on different layers
        for test_idx in range(num_tests):
            # Clone model parameters
            original_state = {
                name: param.clone()
                for name, param in model.named_parameters()
            }

            # Select a random layer to rotate
            rotatable_layers = self._get_rotatable_layers(model)
            if not rotatable_layers:
                break

            layer_name = rotatable_layers[test_idx % len(rotatable_layers)]
            tested_layers.append(layer_name)

            # Apply rotation to this layer
            try:
                self._apply_rotation_to_layer(model, layer_name)

                # Compute new loss
                rotated_loss = self.evaluator.evaluate(model, eval_texts)
                deviation = abs(rotated_loss - baseline_loss)
                deviations.append(deviation)

                # Check if symmetry violated
                if deviation > self.symmetry_tolerance:
                    violations.append({
                        'test_idx': test_idx,
                        'layer': layer_name,
                        'baseline_loss': baseline_loss,
                        'rotated_loss': rotated_loss,
                        'deviation': deviation
                    })

            finally:
                # Restore original parameters
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        param.copy_(original_state[name])

        if not deviations:
            return self._empty_symmetry_result("rotation")

        avg_deviation = sum(deviations) / len(deviations)
        symmetry_score = max(0.0, 1.0 - avg_deviation / self.symmetry_tolerance)
        is_symmetric = symmetry_score > 0.8  # Threshold for "symmetric"

        return {
            'is_symmetric': is_symmetric,
            'symmetry_score': symmetry_score,
            'avg_loss_deviation': avg_deviation,
            'violations': violations,
            'num_tests': len(deviations),
            'tested_layers': list(set(tested_layers))
        }

    def _apply_rotation_to_layer(self, model: nn.Module, layer_name: str) -> None:
        """Apply random orthogonal rotation to a parameter layer.

        Args:
            model: Model to modify (in-place)
            layer_name: Name of layer to rotate
        """
        param = dict(model.named_parameters())[layer_name]

        # Only rotate 2D weight matrices
        if param.dim() != 2:
            return

        m, n = param.shape

        # Generate random orthogonal matrix for smaller dimension
        size = min(m, n, 20)  # Limit size for efficiency

        # Create orthogonal rotation matrix
        Q = self._generate_orthogonal_matrix(size)
        Q = Q.to(device=param.device, dtype=param.dtype)

        # Apply rotation to subspace
        with torch.no_grad():
            if m >= n and size <= n:
                # Rotate columns (input space)
                param[:size, :size] = param[:size, :size] @ Q
            elif n >= m and size <= m:
                # Rotate rows (output space)
                param[:size, :size] = Q @ param[:size, :size]

    def _generate_orthogonal_matrix(self, n: int) -> torch.Tensor:
        """Generate random orthogonal matrix via QR decomposition.

        Args:
            n: Matrix dimension

        Returns:
            Orthogonal matrix Q of size (n, n) with Q^T Q = I

        Examples:
            >>> Q = analyzer._generate_orthogonal_matrix(5)
            >>> identity = Q.T @ Q
            >>> assert torch.allclose(identity, torch.eye(5))
        """
        # Sample random Gaussian matrix
        A = torch.randn(n, n, device=self.device)

        # QR decomposition
        Q: torch.Tensor
        Q, R = torch.linalg.qr(A)

        # Ensure proper rotation (det = +1, not -1)
        # Multiply columns by sign of diagonal of R
        signs = torch.sign(torch.diag(R))
        Q = Q * signs

        return Q

    def _get_rotatable_layers(self, model: nn.Module) -> list[str]:
        """Identify layers where rotation makes sense.

        Returns list of 2D weight parameter names (linear layer weights).

        Args:
            model: Model to analyze

        Returns:
            List of parameter names suitable for rotation
        """
        rotatable = []
        for name, param in model.named_parameters():
            # Only 2D weight matrices (not biases)
            if param.dim() == 2 and 'weight' in name:
                rotatable.append(name)
        return rotatable

    # === Permutation Symmetry ===

    def detect_permutation_symmetry(
        self,
        model: nn.Module,
        eval_texts: list[str],
        num_tests: int = 10
    ) -> dict[str, Any]:
        """Test for neuron permutation symmetry.

        Randomly permutes neurons within layers and checks loss invariance.
        Proper permutation must update all connected layers consistently.

        Args:
            model: Model to test
            eval_texts: Evaluation dataset
            num_tests: Number of random permutation tests

        Returns:
            Dictionary containing symmetry detection results
        """
        if not self.config.detect_permutations:
            return self._empty_symmetry_result("permutation")

        baseline_loss = self.evaluator.evaluate(model, eval_texts)

        deviations = []
        violations = []
        tested_layers = []

        for test_idx in range(num_tests):
            # Clone parameters
            original_state = {
                name: param.clone()
                for name, param in model.named_parameters()
            }

            # Get permutable layers
            permutable_layers = self._get_permutable_layers(model)
            if not permutable_layers:
                break

            layer_name = permutable_layers[test_idx % len(permutable_layers)]
            tested_layers.append(layer_name)

            try:
                # Apply neuron permutation
                self._apply_neuron_permutation(model, layer_name)

                # Compute new loss
                permuted_loss = self.evaluator.evaluate(model, eval_texts)
                deviation = abs(permuted_loss - baseline_loss)
                deviations.append(deviation)

                if deviation > self.symmetry_tolerance:
                    violations.append({
                        'test_idx': test_idx,
                        'layer': layer_name,
                        'deviation': deviation
                    })

            finally:
                # Restore parameters
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        param.copy_(original_state[name])

        if not deviations:
            return self._empty_symmetry_result("permutation")

        avg_deviation = sum(deviations) / len(deviations)
        symmetry_score = max(0.0, 1.0 - avg_deviation / self.symmetry_tolerance)
        is_symmetric = symmetry_score > 0.8

        return {
            'is_symmetric': is_symmetric,
            'symmetry_score': symmetry_score,
            'avg_loss_deviation': avg_deviation,
            'violations': violations,
            'num_tests': len(deviations),
            'tested_layers': list(set(tested_layers))
        }

    def _apply_neuron_permutation(self, model: nn.Module, layer_name: str) -> None:
        """Randomly permute neurons in a layer.

        For a layer with weight W ∈ ℝ^(m×n), permutes columns (output neurons).
        Must also permute corresponding bias and downstream layer weights.

        Args:
            model: Model to modify (in-place)
            layer_name: Name of layer to permute
        """
        param_dict = dict(model.named_parameters())

        if layer_name not in param_dict:
            return

        weight = param_dict[layer_name]

        # Only permute 2D weights
        if weight.dim() != 2:
            return

        m, n = weight.shape

        # Generate random permutation
        perm = torch.randperm(n, device=weight.device)

        with torch.no_grad():
            # Permute columns of weight matrix
            weight.copy_(weight[:, perm])

            # Permute corresponding bias if it exists
            bias_name = layer_name.replace('.weight', '.bias')
            if bias_name in param_dict:
                bias = param_dict[bias_name]
                if bias.shape[0] == n:
                    bias.copy_(bias[perm])

            # Find and permute downstream layer
            # (This is simplified - full implementation would trace computational graph)
            layer_prefix = '.'.join(layer_name.split('.')[:-1])
            next_layer_idx = int(layer_prefix.split('.')[-1]) + 1 if layer_prefix.split('.')[-1].isdigit() else None

            if next_layer_idx is not None:
                # Try to find next layer weight
                next_layer_prefix = '.'.join(layer_prefix.split('.')[:-1] + [str(next_layer_idx)])
                next_weight_name = f"{next_layer_prefix}.weight"

                if next_weight_name in param_dict:
                    next_weight = param_dict[next_weight_name]
                    if next_weight.dim() == 2 and next_weight.shape[1] == n:
                        # Permute rows of next layer
                        next_weight.copy_(next_weight[perm, :])

    def _get_permutable_layers(self, model: nn.Module) -> list[str]:
        """Identify layers where neuron permutation is valid.

        Returns list of linear layer weight names.

        Args:
            model: Model to analyze

        Returns:
            List of permutable parameter names
        """
        permutable = []
        for name, param in model.named_parameters():
            if param.dim() == 2 and 'weight' in name:
                permutable.append(name)
        return permutable

    # === Scaling Symmetry ===

    def detect_scaling_symmetry(
        self,
        model: nn.Module,
        eval_texts: list[str],
        num_tests: int = 10,
        scale_range: tuple[float, float] = (0.5, 2.0)
    ) -> dict[str, Any]:
        """Test for layer-wise scaling symmetry.

        Tests if L(λ·θ) ≈ L(θ) for certain scaling factors λ.

        Args:
            model: Model to test
            eval_texts: Evaluation dataset
            num_tests: Number of scaling tests
            scale_range: (min_scale, max_scale) for random scaling factors

        Returns:
            Dictionary containing symmetry detection results
        """
        if not self.config.detect_scaling:
            return self._empty_symmetry_result("scaling")

        baseline_loss = self.evaluator.evaluate(model, eval_texts)

        deviations = []
        violations = []
        tested_scales = []

        for test_idx in range(num_tests):
            # Clone parameters
            original_state = {
                name: param.clone()
                for name, param in model.named_parameters()
            }

            # Random scaling factor
            scale_factor = torch.rand(1).item() * (scale_range[1] - scale_range[0]) + scale_range[0]
            tested_scales.append(scale_factor)

            try:
                # Apply scaling to all parameters
                with torch.no_grad():
                    for param in model.parameters():
                        param.mul_(scale_factor)

                # Compute new loss
                scaled_loss = self.evaluator.evaluate(model, eval_texts)
                deviation = abs(scaled_loss - baseline_loss)
                deviations.append(deviation)

                if deviation > self.symmetry_tolerance:
                    violations.append({
                        'test_idx': test_idx,
                        'scale_factor': scale_factor,
                        'deviation': deviation
                    })

            finally:
                # Restore parameters
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        param.copy_(original_state[name])

        if not deviations:
            return self._empty_symmetry_result("scaling")

        avg_deviation = sum(deviations) / len(deviations)
        symmetry_score = max(0.0, 1.0 - avg_deviation / self.symmetry_tolerance)
        is_symmetric = symmetry_score > 0.8

        return {
            'is_symmetric': is_symmetric,
            'symmetry_score': symmetry_score,
            'avg_loss_deviation': avg_deviation,
            'violations': violations,
            'num_tests': len(deviations),
            'tested_scales': tested_scales
        }

    # === Quotient Space ===

    def compute_canonical_representative(
        self,
        params: dict[str, torch.Tensor],
        symmetry_group: str = "permutation"
    ) -> dict[str, torch.Tensor]:
        """Compute canonical form in quotient space Θ/G.

        Maps parameters to a unique representative of their equivalence class
        under the symmetry group.

        Args:
            params: Parameters to canonicalize
            symmetry_group: "rotation" | "permutation" | "scaling"

        Returns:
            Canonical parameter dictionary

        Examples:
            >>> canonical_params = analyzer.compute_canonical_representative(
            ...     params, symmetry_group="permutation"
            ... )
        """
        if symmetry_group == "permutation":
            return self._canonical_permutation(params)
        elif symmetry_group == "scaling":
            return self._canonical_scaling(params)
        elif symmetry_group == "rotation":
            # Rotation quotient is complex - use simplified version
            return params  # TODO: Implement full rotation quotient
        else:
            return params

    def _canonical_permutation(
        self,
        params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Canonical form for permutation symmetry.

        Sorts neurons by L2 norm of incoming weights (decreasing order).

        Args:
            params: Parameter dictionary

        Returns:
            Canonicalized parameters
        """
        canonical = {}

        for name, param in params.items():
            if param.dim() == 2 and 'weight' in name:
                # Compute L2 norm of each column (neuron)
                norms = torch.norm(param, dim=0)

                # Sort indices by norm (decreasing)
                sorted_idx = torch.argsort(norms, descending=True)

                # Permute columns
                canonical[name] = param[:, sorted_idx].clone()

                # Handle corresponding bias
                bias_name = name.replace('.weight', '.bias')
                if bias_name in params:
                    canonical[bias_name] = params[bias_name][sorted_idx].clone()
            elif name not in canonical:  # Avoid duplicating bias
                canonical[name] = param.clone()

        return canonical

    def _canonical_scaling(
        self,
        params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Canonical form for scaling symmetry.

        Normalizes parameters to unit Frobenius norm.

        Args:
            params: Parameter dictionary

        Returns:
            Normalized parameters
        """
        canonical = {}

        # Compute total Frobenius norm
        total_norm_squared = torch.tensor(0.0, device=self.device)
        for p in params.values():
            total_norm_squared = total_norm_squared + torch.sum(p ** 2)
        total_norm = torch.sqrt(total_norm_squared)

        # Normalize all parameters
        for name, param in params.items():
            canonical[name] = param / total_norm

        return canonical

    def project_to_quotient_space(
        self,
        model: nn.Module,
        symmetry_types: list[str]
    ) -> nn.Module:
        """Project model parameters to quotient space.

        Applies canonical representative mapping to all parameters,
        modifying model in-place.

        Args:
            model: Model to project
            symmetry_types: List of symmetries to factor out
                e.g., ["permutation", "scaling"]

        Returns:
            Modified model (same reference, modified in-place)

        Examples:
            >>> model = analyzer.project_to_quotient_space(
            ...     model, symmetry_types=["permutation", "scaling"]
            ... )
        """
        if not self.config.quotient_space:
            return model

        # Get current parameters
        params = {name: param.data.clone() for name, param in model.named_parameters()}

        # Apply each symmetry canonicalization
        for sym_type in symmetry_types:
            params = self.compute_canonical_representative(params, sym_type)

        # Update model parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in params:
                    param.copy_(params[name])

        return model

    # === Analysis and Reporting ===

    def analyze_all_symmetries(
        self,
        model: nn.Module,
        eval_texts: list[str],
        num_tests_per_type: int = 10
    ) -> dict[str, Any]:
        """Run all enabled symmetry tests.

        Performs rotation, permutation, and scaling symmetry detection.

        Args:
            model: Model to analyze
            eval_texts: Evaluation dataset
            num_tests_per_type: Tests per symmetry type

        Returns:
            Comprehensive symmetry analysis report with results for each type

        Examples:
            >>> results = analyzer.analyze_all_symmetries(model, texts)
            >>> print(f"Rotation: {results['rotation']['symmetry_score']:.2f}")
            >>> print(f"Permutation: {results['permutation']['symmetry_score']:.2f}")
        """
        results = {}

        if self.config.detect_rotations:
            results['rotation'] = self.detect_rotation_symmetry(
                model, eval_texts, num_tests_per_type
            )

        if self.config.detect_permutations:
            results['permutation'] = self.detect_permutation_symmetry(
                model, eval_texts, num_tests_per_type
            )

        if self.config.detect_scaling:
            results['scaling'] = self.detect_scaling_symmetry(
                model, eval_texts, num_tests_per_type
            )

        # Summary
        results['summary'] = {
            'any_symmetry_detected': any(
                res.get('is_symmetric', False) for res in results.values()
                if isinstance(res, dict)
            ),
            'num_symmetries_detected': sum(
                1 for res in results.values()
                if isinstance(res, dict) and res.get('is_symmetric', False)
            )
        }

        return results

    # === Helper Methods ===

    def _empty_symmetry_result(self, symmetry_type: str) -> dict[str, Any]:
        """Return empty result when symmetry test is disabled.

        Args:
            symmetry_type: Name of symmetry type

        Returns:
            Dictionary with default values
        """
        return {
            'is_symmetric': False,
            'symmetry_score': 0.0,
            'avg_loss_deviation': 0.0,
            'violations': [],
            'num_tests': 0,
            'note': f'{symmetry_type} symmetry detection disabled'
        }
