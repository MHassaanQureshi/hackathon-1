<!--
Sync Impact Report:
- Version change: N/A → 1.0.0
- Added sections: All principles and sections as specified for the AI-Native Book project
- Templates requiring updates: N/A (new constitution)
- Follow-up TODOs: None
-->

# AI-Native Book with Integrated RAG Chatbot Constitution

## Core Principles

### Spec-driven, AI-native development using Claude Code
All development follows specification-driven methodology using Claude Code for AI-assisted development, ensuring accuracy based on official documentation and specs

### Accuracy based on official documentation and specs
All technical claims must be traceable to sources, with no hallucinated content or undocumented behavior allowed

### Clarity for advanced CS / AI audiences
All documentation and code must be clear and comprehensible for advanced computer science and artificial intelligence practitioners

### Reproducibility of content, code, and deployment
All processes must be fully reproducible from repository, following documentation-first workflow using Spec-Kit Plus

### AI-generated output reviewed and corrected
All AI-generated content must be reviewed and corrected by humans before final acceptance

### No hallucinated content or undocumented behavior
All technical content must be grounded in verified sources and documented properly

## Technology Stack and Infrastructure Constraints
Authoring format: Docusaurus (MDX), Deployment: GitHub Pages, Tooling: Claude Code, Spec-Kit Plus, Backend: FastAPI, Databases: Neon Serverless Postgres, Qdrant Cloud (Free Tier), AI stack: OpenAI Agents / ChatKit SDKs

## RAG Chatbot Requirements and Success Criteria
RAG chatbot requirements: Embedded within the published book, Answers grounded only in book content, Supports answering based on user-selected text, Sources always derived from retrieved context. Success criteria: Book successfully deployed on GitHub Pages, RAG chatbot answers accurately without hallucination, System fully reproducible from repository

## Governance
Constitution supersedes all other practices; Amendments require documentation and approval; All PRs/reviews must verify compliance with AI-native development principles, accuracy requirements, and reproducibility standards

**Version**: 1.0.0 | **Ratified**: 2025-12-20 | **Last Amended**: 2025-12-20