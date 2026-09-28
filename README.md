# AI Debate Arena 1.1

Versi 1.1 menambahkan **consensus loop** tanpa judge.

## Mekanisme inti

```text
PERTANYAAN
   ↓
5 AI menjawab independen
   ↓
DEBAT
   ↓
KELEBIHAN / KEKURANGAN / KESAMAAN / PERBEDAAN / KOREKSI
   ↓
KLAIM-KLAIM KONKRET
   ↓
SETIAP AI REVIEW KLAIM
   ↓
┌────────────────────────────────────────────┐
│ ACCEPT dari semua 5 → KONSENSUS           │
│ REJECT/UNCERTAIN satu saja → BELUM SELESAI│
└────────────────────────────────────────────┘
   ↓
Jika belum selesai:
semua AI menerima keberatan + memperbarui argumen
   ↓
review lagi
   ↓
ulang sampai konsensus atau batas ronde
   ↓
POSISI AKHIR MASING-MASING AI
```

### Prinsip penting

Tidak ada judge.

Tidak ada voting.

Tidak ada "4 lawan 1 berarti benar".

Klaim hanya ditandai sebagai **accepted by all** jika kelima model
secara eksplisit memberikan status `ACCEPT`.

`REJECT` atau `UNCERTAIN` dari satu model membuat klaim tetap terbuka
dan dikirim kembali ke loop revisi.

Jika wording sebuah klaim terlalu luas, model dapat meminta penyempitan.
Versi yang lebih sempit kemudian diuji lagi.

## Mengapa ada batas ronde?

Tanpa batas, dua model yang tidak pernah sepakat dapat membuat loop API tanpa
akhir dan menghabiskan biaya. Default 5 consensus rounds, dapat diubah 1–10.

Jika batas tercapai, aplikasi **tidak memalsukan konsensus**. Klaim yang belum
unanimous tetap ditampilkan sebagai unresolved.

## Fitur 1.1

- GPT / Claude / Gemini / Kimi / Grok
- parallel execution
- independent first answer
- multi-round debate
- strengths / weaknesses / similarities / differences / corrections
- candidate claim generation
- per-claim ACCEPT / REJECT / UNCERTAIN
- unanimous consensus gate
- automatic revision loop for unresolved claims
- no judge
- no majority voting
- final position per model
- API failure isolation
- configurable model IDs
- configurable debate rounds
- configurable consensus rounds
- JSON session export
- `.env` secrets

## Instalasi

Python 3.11+ disarankan.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Salin `.env.example` menjadi `.env`, isi API key.

Jalankan:

```bash
streamlit run app.py
```

## Catatan model

Model ID berubah dan akses model tergantung akun/provider. Karena itu ID model
dapat diedit melalui sidebar dan `.env.example`.

Jangan membagikan `.env` atau API key.
