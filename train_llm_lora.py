"""
Llama3 한국어 모델 LoRA 파인튜닝 스크립트
"""
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import TaskType

# 설정
MODEL_NAME = "beomi/Llama-3-Open-Ko-8B"  # 한국어 Llama3 모델
OUTPUT_DIR = "./llm_lora_model"
TRAINING_DATA_PATH = "training_data.json"

def load_training_data(file_path):
    """학습 데이터 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_instruction(sample):
    """Instruction 형식으로 포맷팅"""
    return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""

def prepare_dataset(data, tokenizer):
    """데이터셋 준비"""
    # 텍스트 포맷팅
    texts = [format_instruction(sample) for sample in data]

    # Dataset 생성
    dataset = Dataset.from_dict({"text": texts})

    # 토크나이징
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset

def train_lora():
    """LoRA 파인튜닝"""

    print("=" * 50)
    print("LLM LoRA 파인튜닝 시작")
    print("=" * 50)

    # 1. 학습 데이터 로드
    print(f"\n1. 학습 데이터 로드: {TRAINING_DATA_PATH}")
    training_data = load_training_data(TRAINING_DATA_PATH)
    print(f"   샘플 수: {len(training_data)}")

    # 2. 토크나이저 로드
    print(f"\n2. 토크나이저 로드: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. 데이터셋 준비
    print("\n3. 데이터셋 준비")
    dataset = prepare_dataset(training_data, tokenizer)
    print(f"   토크나이징 완료: {len(dataset)} 샘플")

    # 4. 모델 로드
    print(f"\n4. 모델 로드: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 5. LoRA 설정
    print("\n5. LoRA 설정")
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        target_modules=["q_proj", "v_proj"],  # 적용할 레이어
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # 6. PEFT 모델 준비
    print("\n6. PEFT 모델 준비")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 7. 학습 설정
    print("\n7. 학습 설정")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        save_steps=50,
        logging_steps=10,
        save_total_limit=2,
        warmup_steps=10,
        report_to="none"
    )

    # 8. Trainer 설정
    print("\n8. Trainer 설정")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # 9. 학습 시작
    print("\n9. 학습 시작")
    print("=" * 50)
    trainer.train()

    # 10. 모델 저장
    print("\n10. 모델 저장")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 50)
    print(f"학습 완료! 모델 저장 위치: {OUTPUT_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    train_lora()
