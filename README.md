# ChatGPT Archive Analyzer

A comprehensive suite of tools for analyzing ChatGPT conversation archives, extracting insights, generating visualizations, and performing advanced analytics on AI conversation data.

## 🎯 Features

- **Conversation Analysis**: Statistical analysis of ChatGPT conversations with detailed metrics
- **Data Visualization**: Word clouds, heatmaps, scatter plots, and network graphs
- **Cross-Platform Integration**: Analysis tools for Claude, ChatGPT, and other AI platforms
- **Advanced Analytics**: N-gram analysis, sentiment tracking, and user behavior patterns
- **Export Capabilities**: JSON, XML, HTML, and CSV output formats
- **Real-time Processing**: Batch processing and real-time conversation monitoring

## 📊 Analysis Tools

### Core Analysis Scripts

- **`conversation_analysis.py`** - Primary conversation statistics and metrics
- **`analyzegpt2.py`** - Advanced GPT conversation parser and analyzer
- **`claude_db_advanced.py`** - Claude conversation database analysis
- **`visual_conv_analysis.py`** - Visualization engine for conversation data

### Visualization Tools

- **`gptwordcloud.py`** - Generate word clouds from conversation data
- **`heater2.py`** / **`heater3.py`** - Heatmap generation for activity patterns
- **`graph.py`** - Network graph visualization of conversation flows
- **`plotlongoutput.py`** - Long-form output analysis and plotting

### Cross-Platform Analysis

- **`claude-gptchat-archive-deeper.py`** - Deep cross-platform conversation analysis
- **`claudegptarchiveanalyzer.py`** - Unified Claude/GPT archive processor
- **`analyzed_conversations-duo.py`** - Dual-platform conversation comparison

### Specialized Analytics

- **`anomaly.py`** - Anomaly detection in conversation patterns
- **`vector.py`** - Vector space analysis and embeddings
- **`cluster.py`** - Conversation clustering and topic modeling
- **`fun_analysis.py`** - Entertainment and engagement metrics

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn plotly wordcloud scikit-learn
```

### Basic Usage

1. **Analyze ChatGPT Archives**:
   ```bash
   python conversation_analysis.py --input conversations.json
   ```

2. **Generate Visualizations**:
   ```bash
   python gptwordcloud.py --data conversations.json --output wordcloud.png
   python heater2.py --input user_activity.csv --output heatmap.png
   ```

3. **Cross-Platform Analysis**:
   ```bash
   python claude-gptchat-archive-deeper.py --chatgpt chatgpt.json --claude claude.json
   ```

## 📁 Data Formats

### Input Formats Supported

- **ChatGPT JSON**: Native ChatGPT export format
- **Claude Conversations**: Anthropic Claude conversation logs
- **CSV Data**: Structured conversation data
- **Custom JSON**: User-defined conversation schemas

### Output Formats

- **Statistical Reports**: Detailed analytics in JSON/CSV
- **Visualizations**: PNG, SVG, HTML interactive charts
- **Database Exports**: SQLite, PostgreSQL compatible
- **Web Reports**: HTML dashboards with embedded visualizations

## 🔧 Configuration

### Analysis Parameters

Most scripts support command-line configuration:

```bash
python conversation_analysis.py \
    --input conversations.json \
    --output analysis_report.json \
    --include-metadata \
    --date-range 2024-01-01:2024-12-31 \
    --min-length 50 \
    --export-format json
```

### Visualization Settings

```python
# Example configuration for visualization tools
VISUALIZATION_CONFIG = {
    "wordcloud": {
        "max_words": 100,
        "colormap": "viridis",
        "background_color": "white"
    },
    "heatmap": {
        "resolution": "daily",
        "color_scheme": "RdYlBu"
    }
}
```

## 📈 Analysis Capabilities

### Conversation Metrics

- **Volume Analysis**: Messages per day/week/month
- **Length Statistics**: Character count, word count, response times
- **Topic Tracking**: Theme evolution and topic clustering
- **Engagement Metrics**: Question frequency, follow-up patterns

### User Behavior Analysis

- **Activity Patterns**: Time-of-day and day-of-week usage
- **Session Analysis**: Conversation length and break patterns
- **Query Types**: Question classification and intent analysis
- **Response Quality**: Satisfaction indicators and feedback analysis

### Cross-Platform Insights

- **Platform Comparison**: Usage patterns across different AI services
- **Model Performance**: Response quality and user satisfaction
- **Feature Usage**: Which AI capabilities are most utilized
- **Migration Patterns**: How users move between platforms

## 🎨 Visualization Examples

### Word Clouds
- **Frequency-based**: Most common terms and phrases
- **Sentiment-colored**: Emotional tone visualization
- **Topic-specific**: Domain-focused vocabulary analysis

### Heatmaps
- **Activity Heatmaps**: When users are most active
- **Topic Heatmaps**: What subjects are discussed when
- **Response Quality**: Performance metrics over time

### Network Graphs
- **Conversation Flow**: How topics connect and evolve
- **User Interaction**: Multi-user conversation dynamics
- **Concept Networks**: Related ideas and knowledge mapping

## 🔍 Advanced Features

### Anomaly Detection
```python
python anomaly.py --input conversations.json --threshold 2.5 --output anomalies.json
```

### Vector Analysis
```python
python vector.py --input conversations.json --model sentence-transformers --output embeddings.npy
```

### Clustering
```python
python cluster.py --input conversations.json --method kmeans --clusters 10 --output clusters.json
```

## 📊 Output Examples

### Statistical Report (JSON)
```json
{
  "total_conversations": 1250,
  "total_messages": 15650,
  "average_response_time": "2.3 seconds",
  "top_topics": ["coding", "analysis", "planning"],
  "user_satisfaction": 0.87,
  "platform_distribution": {
    "chatgpt": 0.65,
    "claude": 0.35
  }
}
```

### Visualization Outputs
- **Enhanced heatmaps**: `enhanced_heatmap.png`
- **Function co-occurrence**: `function_co_occurrence_heatmap.png`
- **Interactive graphs**: `function_graph.html`
- **Code analysis**: `code_snippets_pie_chart.png`

## 🛠️ Development

### Adding New Analysis Tools

1. Create analysis script following the pattern:
```python
def analyze_conversations(input_file, output_file, **kwargs):
    # Load data
    # Perform analysis
    # Generate outputs
    pass
```

2. Add configuration options and CLI support
3. Include visualization generation if applicable
4. Add tests and documentation

### Data Pipeline

```
Raw Data → Preprocessing → Analysis → Visualization → Export
    ↓           ↓            ↓            ↓           ↓
  JSON       Clean        Stats      Charts      Reports
  CSV        Filter      Metrics     Graphs      Dashboards
  XML        Transform   Insights    Heatmaps    APIs
```

## 📚 Documentation

### Script Documentation

- **`README-7.md`** - Detailed script documentation
- **`analysis_results_*.txt`** - Sample analysis outputs
- **`conversations.json`** - Example data format

### Generated Reports

- **HTML Dashboards**: Interactive web-based reports
- **Statistical Summaries**: Comprehensive metrics and KPIs
- **Visualization Galleries**: Collection of generated charts and graphs

## 🔒 Privacy & Security

- **Data Anonymization**: Remove personal identifiers before analysis
- **Local Processing**: All analysis performed locally, no data sent to external services
- **Secure Storage**: Encrypted storage options for sensitive conversation data
- **Export Controls**: Configurable data export restrictions

## 🤝 Contributing

This toolkit is designed for researchers, developers, and analysts working with AI conversation data. Contributions welcome for:

- New analysis algorithms
- Additional visualization types
- Platform integrations
- Performance optimizations

## 📄 License

This project is intended for research and educational purposes. Ensure compliance with AI platform terms of service when analyzing conversation data.

## 🙏 Acknowledgments

Built for comprehensive analysis of AI conversation patterns and user behavior insights. Supports research into human-AI interaction, conversation dynamics, and AI system evaluation.