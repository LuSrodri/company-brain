# Arquivos do teste end-to-end

Coloque aqui os arquivos reais usados pelo `tests/test_e2e_ingestion.py`. Cada
caso é ingerido via `POST /documents/upload` e consultado via `POST /chat` com
modelos reais (Gemma 4 + Whisper + harrier).

Arquivos esperados (exatamente estes nomes):

| Modalidade | Arquivo                          | Pergunta                                                                 |
| ---------- | -------------------------------- | ------------------------------------------------------------------------ |
| PDF        | `Data AI Label Project.pdf`      | Do I necessarily need produce hallucination in the process?              |
| TXT        | `Internal Pentesting Report.txt` | What are the CWEs?                                                        |
| Áudio      | `Alinhamento Black Friday.mp3`   | Combos valem para a black friday?                                        |
| DOCX       | `Carta Apresentacao Lucas.docx`  | Qual é a stack do candidato Lucas?                                       |
| XLSX       | `notas.xlsx`                     | Quanto falta para eu passar em Sistemas Operacionais Embarcados?         |
| Imagem     | `Formula 1.png`                  | Como funciona um sábado de um fim de semana de um Grande Prêmio de F1?   |

O teste pula (skip) qualquer caso cujo arquivo não esteja presente, e a suíte
inteira pula se `CB_HF_TOKEN` não estiver definido (em `api/.env.dev` ou no
ambiente). O caso de PDF também exige o **poppler** instalado (ver README da API).

Rode com:

```bash
pytest -m e2e -s
```
