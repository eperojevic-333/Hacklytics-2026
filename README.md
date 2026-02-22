# Hacklytics Project — AI Data Scientist Workflow (Sphinx → Gemini → ElevenLabs)

## Overview
This project is an end-to-end data science workflow that:
1. Ingests data from a source,
2. Uses **Sphinx** as an “AI Data Scientist” to perform analysis and generate findings,
3. Produces an **executive summary** for stakeholders,
4. Sends the deeper analysis to **Gemini** to generate a polished narration/transcription,
5. Passes that script to **ElevenLabs** to generate a **voiceover** that presents and discusses the results.

The result is an automated pipeline that turns raw data into both written insights and an audio-ready presentation.

## Key Features
- **Pluggable data ingestion**: bring your own data source (file, database, API, etc.).
- **AI-driven analysis**: Sphinx performs exploratory analysis and produces structured findings.
- **Executive-ready output**: concise summary generated from the analysis.
- **Narration generation**: Gemini converts technical findings into a presentation-style transcript.
- **Audio voiceover**: ElevenLabs turns the transcript into a natural-sounding voiceover.

## Architecture (High Level)
- **Input**: Data Source  
- **Analysis**: Sphinx (AI Data Scientist)  
- **Outputs**:
  - **Executive Summary** (stakeholder-friendly)
  - **Analysis Report** (technical detail)
- **Narration**: Gemini (transcript from analysis)
- **Audio**: ElevenLabs (voiceover from transcript)

## Workflow Steps
1. **Ingest Data**
   - Load data from the configured source.
2. **Analyze with Sphinx**
   - Run EDA / modeling / insight extraction (depending on configuration).
   - Produce structured results (metrics, trends, charts/tables references, key takeaways).
3. **Generate Executive Summary**
   - Summarize Sphinx’s results into short, decision-oriented points.
4. **Generate Transcript with Gemini**
   - Convert the analysis into a spoken, presentation-style script.
   - Emphasize clarity, pacing, and narrative flow.
5. **Generate Voiceover with ElevenLabs**
   - Produce final audio narration suitable for demos and executive playback.

## Example Outputs
- **Executive Summary**
  - Bullet-point takeaways, risks, and recommendations.
- **Analysis Section**
  - Technical findings, assumptions, supporting evidence.
- **Transcript**
  - A narrated script explaining what was found and why it matters.
- **Voiceover Audio**
  - An MP3/WAV narration presenting the results.

## Tech Stack
- **Sphinx** — AI agent acting as a data scientist (analysis + findings)
- **Gemini** — transcript generation from technical analysis
- **ElevenLabs** — voice synthesis (voiceover)
- **Data Source** — configurable input (CSV/DB/API/etc.)

## How It Works (Conceptually)
- The system treats Sphinx’s output as the “source of truth” for findings.
- From that same output, it branches into:
  - **Executive Summary**: compressed, business-facing conclusions.
  - **Narration Track**: a story-driven transcript for spoken delivery.
- The final audio is generated directly from the transcript to ensure consistent messaging.

## Use Cases
- Automated analytics reporting for non-technical stakeholders
- Demo-friendly project presentations (audio + summary)
- Rapid insight-to-narration workflow for hackathons and internal dashboards

## Future Improvements
- Add evaluation/guardrails for factual consistency between analysis and transcript
- Support multiple voices / tones (formal vs. energetic) for different audiences
- Implement conversational AI to go beyond preseting and into discussing with members
- Develop further to discuss with members in meetings for further application! (Conversational AI)

## Team / Credits
Edi Perojevic
Danny Perojevic
