# Avaliação (evals)

Suíte que **mede a qualidade** do RAG sobre modelos e arquivos reais — o que os
testes unitários (com fakes) não fazem. Produz números para assertividade,
segurança e latência, em vez de só passar/falhar.

## O que é medido

| Dimensão          | Métrica                        | Como é medida                                                                 |
| ----------------- | ------------------------------ | ----------------------------------------------------------------------------- |
| **Assertividade** | Retrieval hit rate             | O trecho esperado aparece nas fontes recuperadas (top-k) para a pergunta.     |
| **Assertividade** | Correção (LLM-as-judge)        | Um 2º Gemini julga se a resposta responde corretamente, dada a referência.    |
| **Assertividade** | Citação de fonte               | A resposta cita arquivo/página/timestamp (guardrail anti-alucinação).         |
| **Segurança**     | Recusa fora de escopo          | Perguntas sem resposta na base → o modelo deve dizer que **não encontrou**.   |
| **Segurança**     | Resistência a prompt injection | Documento com instrução maliciosa embutida → o modelo deve **ignorá-la**.     |
| **Latência**      | Chat p50 / p95                 | Latência server-side (`latency_ms`) de recuperação + geração por chamada.     |
| **Custo/latência**| Ingestão por modalidade        | Tempo de `POST /documents/upload` por tipo de arquivo (VLM/STT no caminho).   |

O dataset (`dataset.py`) tem um caso de assertividade **por modalidade** (pdf,
txt, áudio, docx, xlsx, imagem), quatro perguntas fora de escopo e um documento
com injeção. Ancorado nos mesmos arquivos de [`../tests/e2e_assets/`](../tests/e2e_assets/README.md).

## Rodar

```bash
cd api
python -m evals.run
```

Requisitos (iguais aos do e2e): `CB_GOOGLE_API_KEY`, `CB_OPENAI_API_KEY` e
`CB_HF_TOKEN` no ambiente ou em `api/.env.dev`; os arquivos em `tests/e2e_assets/`;
e o **poppler** instalado (para o caso de PDF). Sem as chaves, o script sai com
código 2 e uma mensagem clara — por isso ele **não** roda no CI de todo push
(consome API paga); rode-o sob demanda e versione o `results/latest.md`.

O relatório é impresso no terminal e salvo em `results/latest.md` (tabela) e
`results/latest.json` (números crus, para diff entre execuções).

## Estrutura

- `dataset.py` — casos de teste (grounded / fora de escopo / injeção).
- `metrics.py` — heurísticas determinísticas + o `LLMJudge`.
- `run.py` — sobe a API real, ingere, roda os casos e escreve o relatório.
