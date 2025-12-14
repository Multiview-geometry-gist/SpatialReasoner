"""Chain-of-Thought (CoT) augmentation for view-aware spatial reasoning.

This module modifies the answer_cot field in training samples to reflect
the viewing angle, enabling the model to learn view-aware reasoning.

Architecture:
    - Template-based CoT modification for efficiency
    - Supports multiple augmentation strategies
    - Preserves core spatial reasoning while adding view context
    - Configurable via YAML

Augmentation Strategies:
    1. prefix: Add view context at the beginning
    2. inline: Insert view references throughout
    3. suffix: Add view summary at the end
    4. full: Complete CoT rewrite with view awareness

Usage:
    from src.data_generation.cot_augmentation import CoTAugmentor

    augmentor = CoTAugmentor()
    modified_cot = augmentor.augment(
        original_cot="Analyzing the scene...",
        view_angle=15.0,
        strategy="prefix"
    )

Configuration:
    See configs/cot_augmentation.yaml for detailed settings.
"""

import re
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CoTAugmentationConfig:
    """Configuration for CoT augmentation.

    Attributes:
        strategy: Augmentation strategy (prefix, inline, suffix, full)
        include_angle_value: Whether to include specific angle values
        angle_precision: Decimal precision for angle values
        view_descriptors: Custom view descriptors by angle range
        preserve_answer: Whether to preserve the final answer unchanged
        random_variation: Add random variation to templates
        seed: Random seed for reproducibility
    """
    strategy: str = "prefix"
    include_angle_value: bool = True
    angle_precision: int = 1
    view_descriptors: Dict[str, str] = field(default_factory=lambda: {
        "large_left": "significantly rotated left",
        "left": "rotated left",
        "slight_left": "slightly rotated left",
        "center": "straight-on",
        "slight_right": "slightly rotated right",
        "right": "rotated right",
        "large_right": "significantly rotated right",
    })
    preserve_answer: bool = True
    random_variation: bool = True
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "CoTAugmentationConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Loaded configuration

        Raises:
            ValueError: If path is invalid or file doesn't exist
        """
        import yaml

        # Validate path
        config_path = Path(path).resolve()
        if not config_path.exists():
            raise ValueError(f"Config file not found: {path}")
        if not config_path.is_file():
            raise ValueError(f"Config path is not a file: {path}")
        if config_path.suffix.lower() not in {'.yaml', '.yml'}:
            raise ValueError(f"Invalid config file extension: {config_path.suffix}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# ============================================================================
# View Descriptor
# ============================================================================

class ViewDescriptor:
    """Generates natural language descriptions of viewing angles.

    This class maps numeric angles to human-readable descriptions,
    supporting multiple verbosity levels and randomized variations.
    """

    # Angle thresholds for categorization
    ANGLE_THRESHOLDS = {
        "large": 45.0,
        "medium": 20.0,
        "slight": 5.0,
    }

    # Template variations for natural language
    ANGLE_TEMPLATES = {
        "formal": [
            "From a {direction} perspective ({angle} degrees)",
            "Viewing from {angle} degrees to the {direction}",
            "With the camera positioned {angle} degrees {direction}",
        ],
        "casual": [
            "Looking at this from the {direction} side",
            "From a {direction} viewpoint",
            "Seeing this from {angle} degrees {direction}",
        ],
        "technical": [
            "At {angle} degree azimuth rotation ({direction})",
            "With {angle} degree {direction} rotation",
            "Camera rotated {angle} degrees {direction}",
        ],
    }

    # Direction descriptors
    DIRECTION_TERMS = {
        "left": ["left", "leftward", "left-hand"],
        "right": ["right", "rightward", "right-hand"],
        "center": ["center", "front", "straight ahead"],
    }

    def __init__(self, config: CoTAugmentationConfig):
        """Initialize the descriptor.

        Args:
            config: CoT augmentation configuration
        """
        self.config = config
        random.seed(config.seed)

    def describe(
        self,
        angle: float,
        style: str = "formal",
        include_angle: Optional[bool] = None,
    ) -> str:
        """Generate a description for the viewing angle.

        Args:
            angle: Rotation angle in degrees (negative = left, positive = right)
            style: Description style (formal, casual, technical)
            include_angle: Override config for including numeric angle

        Returns:
            Natural language description of the view
        """
        if include_angle is None:
            include_angle = self.config.include_angle_value

        # Determine direction
        abs_angle = abs(angle)
        if abs_angle < 1.0:
            direction = "center"
            magnitude = ""
        elif angle < 0:
            direction = "left"
            magnitude = self._get_magnitude(abs_angle)
        else:
            direction = "right"
            magnitude = self._get_magnitude(abs_angle)

        # Format angle value
        angle_str = f"{abs_angle:.{self.config.angle_precision}f}"

        # Get template
        templates = self.ANGLE_TEMPLATES.get(style, self.ANGLE_TEMPLATES["formal"])

        if self.config.random_variation:
            template = random.choice(templates)
            dir_term = random.choice(self.DIRECTION_TERMS.get(direction, [direction]))
        else:
            template = templates[0]
            dir_term = self.DIRECTION_TERMS.get(direction, [direction])[0]

        # Build description
        if direction == "center":
            return "From a straight-on view"

        if include_angle:
            return template.format(direction=dir_term, angle=angle_str)
        else:
            return f"From a {magnitude}{dir_term} perspective"

    def _get_magnitude(self, abs_angle: float) -> str:
        """Get magnitude descriptor for angle.

        Args:
            abs_angle: Absolute angle value

        Returns:
            Magnitude descriptor string
        """
        if abs_angle >= self.ANGLE_THRESHOLDS["large"]:
            return "significantly "
        elif abs_angle >= self.ANGLE_THRESHOLDS["medium"]:
            return ""
        elif abs_angle >= self.ANGLE_THRESHOLDS["slight"]:
            return "slightly "
        else:
            return "very slightly "

    def get_observation_phrase(self, angle: float) -> str:
        """Generate an observation phrase for the angle.

        Args:
            angle: Rotation angle in degrees

        Returns:
            Observation phrase like "from this rotated perspective"
        """
        abs_angle = abs(angle)

        if abs_angle < 1.0:
            phrases = [
                "from this front-facing view",
                "from this direct perspective",
                "looking straight at the scene",
            ]
        elif abs_angle < 15.0:
            phrases = [
                "from this slightly angled view",
                "from this offset perspective",
                "with this minor rotation",
            ]
        elif abs_angle < 45.0:
            phrases = [
                "from this rotated viewpoint",
                "with the camera angled",
                "from this different angle",
            ]
        else:
            phrases = [
                "from this significantly rotated view",
                "with substantial camera rotation",
                "from this extreme angle",
            ]

        if self.config.random_variation:
            return random.choice(phrases)
        return phrases[0]


# ============================================================================
# Augmentation Strategies
# ============================================================================

class AugmentationStrategy(ABC):
    """Base class for CoT augmentation strategies.

    Each strategy defines how to modify the original CoT to include
    view awareness. Strategies are registered and selected via configuration.
    """

    name: str = "base"

    def __init__(self, config: CoTAugmentationConfig):
        """Initialize the strategy.

        Args:
            config: CoT augmentation configuration
        """
        self.config = config
        self.descriptor = ViewDescriptor(config)

    @abstractmethod
    def augment(
        self,
        original_cot: str,
        angle: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Augment the CoT with view awareness.

        Args:
            original_cot: Original chain-of-thought answer
            angle: Viewing angle in degrees
            metadata: Optional additional metadata

        Returns:
            Augmented CoT with view context
        """
        raise NotImplementedError

    def _extract_answer(self, cot: str) -> tuple:
        """Extract the final answer from CoT.

        Args:
            cot: Chain-of-thought text

        Returns:
            Tuple of (main_reasoning, answer_part)
        """
        # Common patterns for answers
        answer_patterns = [
            r"(The answer is [A-D]\.?)",
            r"(Therefore,? the answer is [A-D]\.?)",
            r"(So the answer is [A-D]\.?)",
            r"(Answer: [A-D]\.?)",
            r"([A-D]\s*$)",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, cot, re.IGNORECASE)
            if match:
                answer_part = match.group(1)
                main_reasoning = cot[:match.start()].strip()
                return main_reasoning, answer_part

        # No clear answer pattern found
        return cot, ""

    def _preserve_answer(self, augmented: str, original_answer: str) -> str:
        """Ensure the answer is preserved from original.

        Args:
            augmented: Augmented CoT text
            original_answer: Original answer to preserve

        Returns:
            CoT with original answer preserved
        """
        if not original_answer:
            return augmented

        # Remove any answer from augmented
        for pattern in [
            r"The answer is [A-D]\.?",
            r"Therefore,? the answer is [A-D]\.?",
        ]:
            augmented = re.sub(pattern, "", augmented, flags=re.IGNORECASE)

        augmented = augmented.strip()
        if not augmented.endswith("."):
            augmented += "."

        return f"{augmented} {original_answer}"


class PrefixStrategy(AugmentationStrategy):
    """Add view context at the beginning of the CoT.

    This strategy prepends a view-aware introduction to the original CoT,
    establishing the viewing context before the main reasoning.

    Example:
        Original: "Looking at the objects in the image..."
        Augmented: "From a 15-degree right rotation, I can observe the scene
                   from a different angle. Looking at the objects in the image..."
    """

    name = "prefix"

    # Prefix templates
    TEMPLATES = [
        "{view_desc}, I can observe the spatial relationships in this scene. {original}",
        "{view_desc}, the arrangement of objects becomes clearer. {original}",
        "Analyzing the scene {observation}: {original}",
        "{view_desc}, let me examine the spatial layout. {original}",
    ]

    def augment(
        self,
        original_cot: str,
        angle: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add view context prefix to CoT.

        Args:
            original_cot: Original chain-of-thought
            angle: Viewing angle in degrees
            metadata: Optional metadata

        Returns:
            CoT with view context prefix
        """
        main_reasoning, answer_part = self._extract_answer(original_cot)

        view_desc = self.descriptor.describe(angle)
        observation = self.descriptor.get_observation_phrase(angle)

        if self.config.random_variation:
            template = random.choice(self.TEMPLATES)
        else:
            template = self.TEMPLATES[0]

        augmented = template.format(
            view_desc=view_desc,
            observation=observation,
            original=main_reasoning,
        )

        if self.config.preserve_answer and answer_part:
            return self._preserve_answer(augmented, answer_part)

        return augmented


class InlineStrategy(AugmentationStrategy):
    """Insert view references throughout the CoT.

    This strategy modifies the CoT by inserting view-aware phrases
    at relevant points, making the reasoning more naturally view-aware.

    Example:
        Original: "The red ball is to the left of the blue cube."
        Augmented: "From this 15-degree rotation, the red ball appears to the
                   left of the blue cube from my current viewpoint."
    """

    name = "inline"

    # Phrases to enhance with view awareness
    SPATIAL_TRIGGERS = [
        (r"\b(is|appears?)\s+(to the\s+)?(left|right|above|below|in front|behind)",
         "from this angle, \\g<0>"),
        (r"\b(can see|observe|notice)\s+",
         "\\g<0>from this viewpoint "),
        (r"\b(looking at|examining)\s+",
         "\\g<0>from this perspective "),
    ]

    def augment(
        self,
        original_cot: str,
        angle: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert view references inline in CoT.

        Args:
            original_cot: Original chain-of-thought
            angle: Viewing angle in degrees
            metadata: Optional metadata

        Returns:
            CoT with inline view references
        """
        main_reasoning, answer_part = self._extract_answer(original_cot)

        # Add view context prefix
        view_desc = self.descriptor.describe(angle)
        augmented = f"{view_desc}, I analyze the scene. {main_reasoning}"

        # Apply inline modifications (limit to avoid over-modification)
        modifications_made = 0
        max_modifications = 2

        for pattern, replacement in self.SPATIAL_TRIGGERS:
            if modifications_made >= max_modifications:
                break

            # Only modify first occurrence
            new_text, count = re.subn(pattern, replacement, augmented, count=1)
            if count > 0:
                augmented = new_text
                modifications_made += 1

        if self.config.preserve_answer and answer_part:
            return self._preserve_answer(augmented, answer_part)

        return augmented


class SuffixStrategy(AugmentationStrategy):
    """Add view summary at the end of the CoT.

    This strategy appends a view-aware conclusion that summarizes
    how the viewing angle affected the observation.

    Example:
        Original: "The answer is A."
        Augmented: "The answer is A. This conclusion was reached by analyzing
                   the scene from a 15-degree rotated viewpoint."
    """

    name = "suffix"

    # Suffix templates
    TEMPLATES = [
        "{reasoning} {answer} This analysis considered the spatial relationships visible {observation}.",
        "{reasoning} {answer} The observation was made {view_desc}.",
        "{reasoning} {answer} Note: This reasoning accounts for the {angle_desc} viewing angle.",
    ]

    def augment(
        self,
        original_cot: str,
        angle: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add view context suffix to CoT.

        Args:
            original_cot: Original chain-of-thought
            angle: Viewing angle in degrees
            metadata: Optional metadata

        Returns:
            CoT with view context suffix
        """
        main_reasoning, answer_part = self._extract_answer(original_cot)

        observation = self.descriptor.get_observation_phrase(angle)
        view_desc = self.descriptor.describe(angle)

        abs_angle = abs(angle)
        if abs_angle < 5:
            angle_desc = "frontal"
        elif abs_angle < 20:
            angle_desc = "slightly rotated"
        elif abs_angle < 45:
            angle_desc = "moderately rotated"
        else:
            angle_desc = "significantly rotated"

        if self.config.random_variation:
            template = random.choice(self.TEMPLATES)
        else:
            template = self.TEMPLATES[0]

        augmented = template.format(
            reasoning=main_reasoning,
            answer=answer_part or "",
            observation=observation,
            view_desc=view_desc,
            angle_desc=angle_desc,
        )

        return augmented.strip()


class FullStrategy(AugmentationStrategy):
    """Complete CoT rewrite with comprehensive view awareness.

    This strategy generates a new CoT that fully integrates view awareness
    throughout the reasoning process. It uses templates that maintain
    the original spatial reasoning while adding explicit view context.

    This is the most thorough but also most intrusive strategy.
    """

    name = "full"

    # Full rewrite template
    TEMPLATE = """Analyzing this scene {view_desc}:

{view_context}

{original_reasoning}

{view_conclusion}

{answer}"""

    VIEW_CONTEXT_TEMPLATES = [
        "From this viewing angle, I can observe the spatial arrangement of objects in the scene.",
        "This perspective allows me to see the relative positions of the elements.",
        "The current viewpoint reveals the spatial relationships between objects.",
    ]

    VIEW_CONCLUSION_TEMPLATES = [
        "Taking into account the {angle_desc} viewing angle",
        "Considering this {angle_desc} perspective",
        "Based on the observations from this {angle_desc} viewpoint",
    ]

    def augment(
        self,
        original_cot: str,
        angle: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Completely rewrite CoT with view awareness.

        Args:
            original_cot: Original chain-of-thought
            angle: Viewing angle in degrees
            metadata: Optional metadata

        Returns:
            Fully rewritten CoT with view awareness
        """
        main_reasoning, answer_part = self._extract_answer(original_cot)

        view_desc = self.descriptor.describe(angle)

        abs_angle = abs(angle)
        if abs_angle < 5:
            angle_desc = "frontal"
        elif abs_angle < 20:
            angle_desc = "slightly angled"
        elif abs_angle < 45:
            angle_desc = "rotated"
        else:
            angle_desc = "significantly rotated"

        if self.config.random_variation:
            view_context = random.choice(self.VIEW_CONTEXT_TEMPLATES)
            view_conclusion = random.choice(self.VIEW_CONCLUSION_TEMPLATES)
        else:
            view_context = self.VIEW_CONTEXT_TEMPLATES[0]
            view_conclusion = self.VIEW_CONCLUSION_TEMPLATES[0]

        augmented = self.TEMPLATE.format(
            view_desc=view_desc,
            view_context=view_context,
            original_reasoning=main_reasoning,
            view_conclusion=view_conclusion.format(angle_desc=angle_desc),
            answer=answer_part or "",
        )

        # Clean up whitespace
        augmented = re.sub(r'\n{3,}', '\n\n', augmented)

        return augmented.strip()


# ============================================================================
# Strategy Registry
# ============================================================================

class StrategyRegistry:
    """Registry for augmentation strategies.

    Allows dynamic registration and selection of strategies.
    """

    _strategies: Dict[str, type] = {}

    @classmethod
    def register(cls, strategy_cls: type) -> type:
        """Register a strategy class.

        Args:
            strategy_cls: Strategy class to register

        Returns:
            The registered class (for decorator use)
        """
        cls._strategies[strategy_cls.name] = strategy_cls
        return strategy_cls

    @classmethod
    def get(cls, name: str) -> type:
        """Get a strategy class by name.

        Args:
            name: Strategy name

        Returns:
            Strategy class

        Raises:
            KeyError: If strategy not found
        """
        if name not in cls._strategies:
            raise KeyError(
                f"Strategy '{name}' not found. "
                f"Available: {list(cls._strategies.keys())}"
            )
        return cls._strategies[name]

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all registered strategy names."""
        return list(cls._strategies.keys())


# Register built-in strategies
StrategyRegistry.register(PrefixStrategy)
StrategyRegistry.register(InlineStrategy)
StrategyRegistry.register(SuffixStrategy)
StrategyRegistry.register(FullStrategy)


# ============================================================================
# Main Augmentor
# ============================================================================

class CoTAugmentor:
    """Main class for augmenting CoT answers with view awareness.

    This is the primary interface for CoT augmentation. It handles:
    - Strategy selection and instantiation
    - Batch processing of samples
    - Configuration management

    Example:
        augmentor = CoTAugmentor(strategy="prefix")

        # Single sample
        modified_cot = augmentor.augment(
            original_cot="Looking at the objects...",
            view_angle=15.0
        )

        # Batch processing
        augmented_samples = augmentor.augment_batch(samples)
    """

    def __init__(
        self,
        config: Optional[CoTAugmentationConfig] = None,
        strategy: Optional[str] = None,
    ):
        """Initialize the augmentor.

        Args:
            config: Configuration object (uses defaults if None)
            strategy: Override strategy name from config
        """
        self.config = config or CoTAugmentationConfig()

        strategy_name = strategy or self.config.strategy
        strategy_cls = StrategyRegistry.get(strategy_name)
        self.strategy = strategy_cls(self.config)

        logger.debug(f"Initialized CoTAugmentor with strategy: {strategy_name}")

    def augment(
        self,
        original_cot: str,
        view_angle: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Augment a single CoT answer with view awareness.

        Args:
            original_cot: Original chain-of-thought answer
            view_angle: Viewing angle in degrees
            metadata: Optional additional metadata

        Returns:
            Augmented CoT with view context
        """
        return self.strategy.augment(original_cot, view_angle, metadata)

    def augment_sample(
        self,
        sample: Dict[str, Any],
        view_angle: float,
    ) -> Dict[str, Any]:
        """Augment a complete sample dictionary.

        Args:
            sample: Sample dictionary with 'answer_cot' field
            view_angle: Viewing angle for this sample

        Returns:
            Modified sample with augmented 'answer_cot'
        """
        augmented = sample.copy()

        if "answer_cot" in sample:
            augmented["answer_cot"] = self.augment(
                sample["answer_cot"],
                view_angle,
                metadata=sample,
            )

        # Add view metadata
        augmented["view_angle"] = view_angle
        augmented["cot_augmented"] = True
        augmented["augmentation_strategy"] = self.strategy.name

        return augmented

    def augment_batch(
        self,
        samples: List[Dict[str, Any]],
        view_key: str = "selected_angle",
    ) -> List[Dict[str, Any]]:
        """Augment a batch of samples.

        Args:
            samples: List of sample dictionaries
            view_key: Key containing view angle in each sample

        Returns:
            List of augmented samples
        """
        augmented = []
        for sample in samples:
            view_angle = sample.get(view_key, 0.0)
            augmented.append(self.augment_sample(sample, view_angle))
        return augmented


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for CoT augmentation testing."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Test CoT augmentation with different strategies"
    )
    parser.add_argument(
        "--cot",
        type=str,
        default="Looking at the objects in the image, I can see a red ball positioned to the left of a blue cube. The ball appears to be closer to the camera. Therefore, the answer is A.",
        help="Original CoT to augment",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=15.0,
        help="View angle in degrees",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=StrategyRegistry.list_strategies(),
        default="prefix",
        help="Augmentation strategy",
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Show output for all strategies",
    )

    args = parser.parse_args()

    if args.all_strategies:
        print(f"Original CoT:\n{args.cot}\n")
        print(f"View Angle: {args.angle} degrees\n")
        print("=" * 60)

        for strategy_name in StrategyRegistry.list_strategies():
            augmentor = CoTAugmentor(strategy=strategy_name)
            result = augmentor.augment(args.cot, args.angle)

            print(f"\nStrategy: {strategy_name.upper()}")
            print("-" * 40)
            print(result)
            print()

    else:
        augmentor = CoTAugmentor(strategy=args.strategy)
        result = augmentor.augment(args.cot, args.angle)

        print(f"Strategy: {args.strategy}")
        print(f"View Angle: {args.angle}")
        print("-" * 40)
        print(result)


if __name__ == "__main__":
    main()
