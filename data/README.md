# SITV Evaluation Datasets

This directory contains all training and evaluation datasets for SITV experiments.

## Directory Structure

```
data/
├── README.md                    # This file
├── general/                     # General language modeling evaluation
│   ├── wikitext.txt            # Wikipedia-style factual text
│   ├── common_knowledge.txt    # General world knowledge
│   ├── coding.txt              # Code snippets across languages
│   └── mixed_domain.txt        # Various domains mixed
├── tasks/                       # Task-specific training data
│   ├── sentiment_positive.txt
│   ├── sentiment_negative.txt
│   ├── instruction_following.txt
│   └── qa_factual.txt
└── eval/                        # Task-specific evaluation data
    ├── sentiment_positive_eval.txt
    ├── sentiment_negative_eval.txt
    ├── instruction_following_eval.txt
    └── qa_factual_eval.txt
```

## File Format

All dataset files use a simple plain text format:

- **One example per line**
- **Empty lines are ignored**
- **Lines starting with `#` are comments**
- **UTF-8 encoding**

Example:
```
# This is a comment
This is the first training example.
This is the second training example.

# Empty lines above are ignored
This is the third example.
```

## Dataset Types

### 1. General Evaluation (`data/general/`)

These datasets measure the model's **general language modeling capability** across various domains. They are used to understand how task vectors affect the model beyond just the specific task they were trained on.

**Purpose**: Answer the question "Does the task vector degrade general capabilities?"

**Key properties**:
- Domain-diverse (technical, conversational, factual, creative)
- Not task-specific
- Representative of broad language understanding

### 2. Task Training (`data/tasks/`)

These datasets contain training examples for specific tasks. Each file corresponds to one task and will be repeated according to `data_repetition_factor` in the config.

**Naming convention**: `{task_name}.txt`

**Typical size**: 30+ unique examples

### 3. Task Evaluation (`data/eval/`)

These datasets contain **held-out** evaluation examples for each task. They should NOT overlap with training data and should test the same skill/domain.

**Naming convention**: `{task_name}_eval.txt`

**Typical size**: 15-20 examples

## Usage in Config

Specify which general evaluation dataset to use in `config.yaml`:

```yaml
evaluation:
  general_dataset: "mixed_domain"  # Options: mixed_domain, wikitext, coding, common_knowledge, combined
  # Use "combined" to evaluate on all general datasets together
```

### Dataset Options

- **`mixed_domain`** - Diverse multi-domain content (30 examples)
- **`wikitext`** - Wikipedia-style factual content (30 examples)
- **`coding`** - Programming and technical content (30 examples)
- **`common_knowledge`** - Everyday general knowledge (30 examples)
- **`combined`** - All general datasets combined (120 examples total - most comprehensive)

## Creating New Datasets

1. **Choose appropriate directory** (`general/`, `tasks/`, or `eval/`)
2. **Create `.txt` file** with descriptive name
3. **Add examples** (one per line)
4. **Add comments** (optional, for organization)
5. **Update config** if needed

## Best Practices

- **Keep general evaluation balanced** across domains
- **Avoid train/eval overlap** for task-specific data
- **Use realistic examples** that test the intended capability
- **Document special formats** if using structured data
- **Version control** all datasets

## Example Workflow

```python
from sitv.data.loader import DatasetLoader

loader = DatasetLoader()

# Load general evaluation data
general_eval = loader.load_general("mixed_domain")

# Load task training data
train_data = loader.load_task("sentiment_positive")

# Load task evaluation data
task_eval = loader.load_eval("sentiment_positive_eval")
```
