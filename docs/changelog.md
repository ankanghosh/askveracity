# Changelog

All notable changes to the AskVeracity fact-checking and misinformation detection system will be documented in this file.

## [0.4.3] - 2025-06-15

### Added
- Integrated detailed performance tracking for evidence retrieval:
  - Logged source-wise success/failure counts for `RSS`, `Wikidata`, and `OpenAlex`
  - Captured confidence scores and processing durations in `agent.py`
- Added a **Confidence Note** message in the app UI to clarify that the displayed confidence percentage reflects overall verdict certainty, whereas individual evidence may vary in confidence based on source reliability

### Changed
- Refactored multiple `try-except` blocks in `evidence_retrieval.py` to ensure logging of both successful and failed retrieval attempts
- Enhanced fallback mechanism for RSS feeds:
  - AI category now always uses both `AI-specific` and `Technology` RSS feeds
  - Updated sampling logic to avoid negative values
- Improved robustness in logging:
  - Used `.get()` with defaults when accessing dictionary keys in logs to avoid `KeyError`
- Cleaned up imports across all modules to remove redundancy and optimize performance
- Moved module-level imports from function body to the top across several files (e.g., `agent.py`, `app.py`, `models.py`)
- Refactored confidence logging in `truth_classifier()` and `process_claim()` for centralized tracking

### Removed
- Unused or redundant imports: `langdetect`, `spacy`, `re`, `json`, `ssl`, `sys`, `Timer`, etc.
- `PerformanceTracker` initialization in files where it is no longer required (`classification.py`, `evaluate_performance.py`)

## [0.4.2] - 2025-04-28

### Added
- Added performance metrics (Accuracy: 50.0%-57.5%, Safety Rate: 82.5%-85.0%) to app's About section

### Changed
- Updated claim examples in app.py input placeholder
- Updated app_screenshot.png to reflect current UI changes

## [0.4.1] - 2025-04-25

### Updated
- Updated architecture.md to improve accuracy of system description
- Updated README.md to better reflect current system functionality
- Removed references to deprecated source credibility assessment
- Clarified documentation of domain quality boost in RSS feed processing

## [0.4.0] - 2025-04-24

### Added
- Added safety rate metric to performance evaluation
  - Measures how often the system avoids making incorrect assertions
  - Tracks when system correctly abstains from judgment by using "Uncertain" 
  - Included in overall metrics and per-class metrics
- New safety rate visualization chart in performance evaluation
- Added safety flag to detailed claim results

### Updated
- Enhanced `evaluate_performance.py` script to track and calculate safety rates
- Updated documentation to explain the safety rate metric and its importance
- Improved tabular display of performance metrics with safety rate column

## [0.3.0] - 2025-04-23

### Added
- Performance evaluation script (`evaluate_performance.py`) in root directory
- Performance results visualization and storage in `results/` directory
- Enhanced error handling and fallback mechanisms
- Refined relevance scoring with entity and verb matching with keyword fallback for accurate evidence assessment
- Enhanced evidence relevance with weighted scoring prioritization and increased gathering from 5 to 10 items
- Added detailed confidence calculation for more reliable verdicts with better handling of low confidence cases
- Category-specific RSS feeds for more targeted evidence retrieval
- OpenAlex integration for scholarly evidence (replacing Semantic Scholar)

### Changed
- Improved classification output structure for consistent downstream processing
- Added fallback mechanisms for explanation generation and classification
- Improved evidence retrieval and classification mechanism
- Streamlined architecture by removing source credibility and semantic analysis complexity
- Improved classification mechanism with weighted evidence count (55%) and quality (45%)
- Updated documentation to reflect the updated performance metrics, enhanced evidence processing pipeline, improved classification mechanism, and streamlined architecture

### Fixed
- Enhanced handling of non-standard response formats

## [0.2.0] - 2025-04-22

### Added
- Created comprehensive documentation in `/docs` directory
  - `architecture.md` for system design and component interactions
  - `configuration.md` for setup and environment configuration
  - `data-handling.md` for data processing and flow
  - `changelog.md` for version history tracking
- Updated app description to emphasize misinformation detection capabilities

### Changed
- Improved directory structure with documentation folder
- Enhanced README with updated project structure
- Clarified misinformation detection focus in documentation

## [0.1.0] - 2025-04-21

### Added
- Initial release of AskVeracity fact-checking system
- Streamlit web interface in `app.py`
- LangGraph ReAct agent implementation in `agent.py`
- Multi-source evidence retrieval system
  - Wikipedia integration
  - Wikidata integration
  - News API integration
  - RSS feed processing
  - Google's FactCheck Tools API integration
  - OpenAlex scholarly evidence
- Truth classification with LLM
- Explanation generation
- Performance tracking utilities
- Rate limiting and API error handling
- Category detection for source prioritization

### Features
- User-friendly claim input interface
- Detailed results display with evidence exploration
- Category-aware source prioritization
- Robust error handling and fallbacks
- Parallel evidence retrieval for improved performance
- Support for various claim categories:
  - AI
  - Science
  - Technology
  - Politics
  - Business
  - World news
  - Sports
  - Entertainment

## Unreleased

### Planned Features
- Enhanced visualization of evidence relevance
- Display agent reasoning process for greater transparency
- Support for user feedback on verification results
- Streamlined fact-checking using only relevant sources
- Source weighting for improved result relevance
- Improved verdict confidence for challenging / ambiguous claims
- Expanded fact-checking sources
- Improved handling of multilingual claims
- Integration with additional academic databases
- Custom source credibility configuration interface
- Historical claim verification database
- API endpoint for programmatic access