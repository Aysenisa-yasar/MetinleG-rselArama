# Turkce Gorsel Arama

Turkce metin sorgulari ile `atasoglu/flickr8k-turkish` veri seti uzerinde image retrieval yapan moduler bir Python/PyTorch projesidir. Baseline cozum, CLIP'in ViT tabanli gorsel encoder'ini ve Turkce dahil cok dilli metinleri ortak embedding uzayina tasiyan `sentence-transformers/clip-ViT-B-32-multilingual-v1` modelini kullanir.

## Neden bu veri seti?

`atasoglu/flickr8k-turkish`, Turkce image-caption eslesmeleri iceren hazir bir benchmark sunar. Projede retrieval mantigi oncelikli oldugu icin, her gorselin birden fazla Turkce aciklamaya sahip olmasi hem niteliksel arama hem de Recall@K temelli degerlendirme icin uygundur.

## Neden CNN yerine ViT / CLIP?

Bu projede CNN kullanilmamistir. Gorsel tarafta, CLIP ailesinin ViT-B/32 tabanli transformer encoder'i kullanilir. Boylece hem istenen teknik kisit korunur hem de metin-gorsel ortak embedding uzayi icinde retrieval yapmak kolaylasir.

## Proje Yapisi

```text
turkce-gorsel-arama/
|
|- data/
|  |- raw/
|  |- processed/
|  \- embeddings/
|- notebooks/
|  \- exploration.ipynb
|- src/
|  |- __init__.py
|  |- app.py
|  |- config.py
|  |- data_loader.py
|  |- embedder.py
|  |- evaluate.py
|  |- metrics.py
|  |- preprocess.py
|  |- retrieval.py
|  |- train.py
|  \- visualize.py
|- outputs/
|  |- checkpoints/
|  |- figures/
|  |- logs/
|  \- reports/
|- requirements.txt
|- README.md
\- run_pipeline.py
```

## Kurulum

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Kullanilan Veri Seti ve Modeller

- Veri seti: `atasoglu/flickr8k-turkish`
- Splitler: `train=6000`, `validation=1000`, `test=1000`
- Gorsel encoder: `clip-ViT-B-32`
- Metin encoder: `sentence-transformers/clip-ViT-B-32-multilingual-v1`
- Benzerlik olcumu: cosine similarity

Baseline yapida test split'indeki tum gorseller icin image embedding uretilir. Her Turkce caption veya kullanici sorgusu ayni uzaya encode edilir ve benzerlik skoruna gore top-k sonuc listelenir.

## Veri Hazirlama ve Calistirma Komutlari

### 1. Veri indirme / hazirlama

```bash
python run_pipeline.py --stage prepare_data
```

Bu adim:

- Hugging Face uzerinden veri setini indirir
- kolon yapisini inceler
- caption alanlarini tespit eder
- gorselleri `data/raw/<split>/` altina kaydeder
- standart kayitlari `data/processed/` altina yazar

### 2. Image embedding cikarma

```bash
python run_pipeline.py --stage embed
```

Bu adim:

- `train`, `validation`, `test` splitleri icin image embedding uretir
- embeddingleri `data/embeddings/<split>_image_embeddings.pt` olarak kaydeder
- metadata'yi `data/embeddings/<split>_image_metadata.csv` olarak kaydeder

### 3. Baseline evaluation

```bash
python run_pipeline.py --stage evaluate
```

Bu adim:

- test split'indeki her caption'i sorgu olarak kullanir
- dogru gorselin rank'ini bulur
- Recall@1, Recall@5, Recall@10, MRR, MeanRank ve MedianRank hesaplar
- raporlari `outputs/reports/` altina yazar
- gorsellestirmeleri `outputs/figures/` altina kaydeder

### 4. Tum baseline pipeline

```bash
python run_pipeline.py --stage all
```

`all` sirayla `prepare_data -> embed -> evaluate` calistirir.

### 5. Fine-tuning

```bash
python run_pipeline.py --stage train
```

Bu asama:

- baseline train embeddingleri uzerinden hard negative mining yapar
- text encoder'in son katmanlarini ve CLIP vision tower'in son ViT bloklarini acarak joint fine-tuning uygular
- in-batch contrastive loss + explicit hard-negative ranking loss kullanir
- validation split'te retrieval metrigi ile en iyi checkpoint'i secer
- best checkpoint ile test split'ini yeniden embed edip degerlendirir
- checkpointleri `outputs/checkpoints/` altina yazar

### 6. Demo arayuzu

```bash
streamlit run src/app.py
```

Arayuzde kullanici Turkce sorgu girer ve secilen split icin top-k gorseller similarity skorlari ile listelenir. Sidebar uzerinden `baseline` veya `finetuned` model varyanti ve `exact` ya da `ann_lsh` retrieval backend'i secilebilir.

## Metrikler Ne Anlama Geliyor?

- `Recall@1`: Dogru gorsel ilk sirada mi?
- `Recall@5`: Dogru gorsel ilk 5 sonuc icinde mi?
- `Recall@10`: Dogru gorsel ilk 10 sonuc icinde mi?
- `MRR`: Dogru sonucun ust siralarda gelmesini odullendiren ortalama reciprocal rank
- `MeanRank`: Dogru gorselin ortalama sirasi
- `MedianRank`: Dogru gorselin medyan sirasi

Bu proje siniflandirma degil retrieval problemi oldugu icin `accuracy` ana metrik olarak kullanilmaz.

## Beklenen Cikti Dosyalari

### Veri

- `data/processed/dataset_schema.json`
- `data/processed/train_records.jsonl`
- `data/processed/validation_records.jsonl`
- `data/processed/test_records.jsonl`

### Embedding

- `data/embeddings/train_image_embeddings.pt`
- `data/embeddings/validation_image_embeddings.pt`
- `data/embeddings/test_image_embeddings.pt`
- `data/embeddings/train_image_metadata.csv`
- `data/embeddings/validation_image_metadata.csv`
- `data/embeddings/test_image_metadata.csv`

### Raporlar

- `outputs/reports/dataset_summary.txt`
- `outputs/reports/test_metrics.csv`
- `outputs/reports/test_summary.txt`
- `outputs/reports/test_detailed_results.csv`
- `outputs/reports/test_error_analysis.csv`
- `outputs/reports/test_error_summary.txt`
- `outputs/reports/test_sample_queries.csv`
- `outputs/reports/finetuned_best_test_metrics.csv`
- `outputs/reports/finetuned_best_test_summary.txt`
- `outputs/reports/finetuned_best_test_detailed_results.csv`
- `outputs/reports/finetuned_best_test_error_analysis.csv`
- `outputs/reports/finetuned_best_test_error_summary.txt`
- `outputs/reports/finetuned_best_comparison_report.txt`
- `outputs/reports/finetuned_best_training_history.csv`

### Gorseller

- `outputs/figures/test_recall_at_k.png`
- `outputs/figures/test_rank_distribution.png`
- `outputs/figures/test_successful_examples.png`
- `outputs/figures/test_failed_examples.png`
- `outputs/figures/test_<query>_top5.png`
- `outputs/figures/finetuned_best_test_recall_at_k.png`
- `outputs/figures/finetuned_best_test_rank_distribution.png`

## Kod Tasarimi

- `src/data_loader.py`: veri setini indirir, kolonlari inceler ve standart formata cevirir
- `src/preprocess.py`: caption temizleme ve caption flatten islemleri
- `src/embedder.py`: image/text embedding uretimi ve kaydi
- `src/retrieval.py`: cosine similarity ile top-k arama
- `src/evaluate.py`: retrieval evaluation ve raporlama
- `src/visualize.py`: top-k, recall ve basari/basarisizlik gorselleri
- `src/train.py`: hard negative mining, partial text/vision unfreezing ve contrastive fine-tuning
- `src/app.py`: Streamlit demo

## Notlar

- GPU varsa otomatik kullanilir, yoksa CPU ile devam edilir.
- Tum path yonetimi `pathlib` ile yapilir.
- Seed sabitlenmistir.
- Ilk asama temiz bir baseline'dir; production optimizasyonu hedeflenmemistir.
- Retrieval tarafinda `exact` aramaya ek olarak hafif bir `ann_lsh` backend secenegi vardir.
- Fine-tuning sonrasi test split icin ayri embedding ve rapor dosyalari uretilir.

## Olası Iyilestirmeler

- Fine-tuning sirasinda vision tower'in bir kismini kontrollu bicimde acmak
- Hard negative mining eklemek
- Daha buyuk cok dilli CLIP varyantlarini denemek
- FAISS benzeri bir ANN index ile arama hizini artirmak
- Daha zengin hata analizi ve caption-level raporlama eklemek
