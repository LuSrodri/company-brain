"""Suíte de avaliação (evals) do Company Brain.

Mede — sobre modelos e arquivos reais — a **assertividade** (recuperação +
correção via LLM-as-judge + citação de fonte), a **segurança** (taxa de recusa
fora do contexto e resistência a prompt injection) e a **latência** (p50/p95) do
pipeline de RAG. Rode com ``python -m evals.run`` (a partir de ``api/``).
"""
