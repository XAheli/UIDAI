# Research Paper

## Comprehensive Statistical and Machine Learning Analysis of UIDAI Aadhaar Enrollment Data

### Authors
1. **Shuvam Banerji Seal** - Lead Researcher, Data Science and Analytics
2. **Alok Mishra** - Co-Researcher, Statistical Analysis  
3. **Aheli Poddar** - Co-Researcher, Machine Learning and Visualization

### Files

| File | Description |
|------|-------------|
| `research_paper.md` | Full paper in Markdown format |
| `research_paper.tex` | LaTeX version for academic submission |
| `references.bib` | BibTeX bibliography file |

### Abstract

This paper presents a comprehensive statistical and machine learning analysis of the UIDAI Aadhaar enrollment dataset comprising over 6.1 million records across biometric (3.5M), demographic (1.6M), and enrollment (982K) datasets. The study spans 36 states and union territories, approximately 960 districts, with 26 attributes per record.

### Key Findings

1. **Temporal Patterns**: Clear weekday-weekend distinction with Tuesday showing peak enrollment
2. **Geographic Disparities**: Significant regional imbalance with Northeast underrepresented
3. **Socioeconomic Correlations**: Strong positive correlation between HDI, literacy and enrollment
4. **ML Performance**: Random Forest achieved 100% classification accuracy

### Compiling the LaTeX Version

```bash
# Install LaTeX (if not already installed)
# Ubuntu/Debian: sudo apt install texlive-full
# macOS: brew install mactex

# Compile the paper
cd paper/
pdflatex research_paper.tex
bibtex research_paper
pdflatex research_paper.tex
pdflatex research_paper.tex
```

### Citation

If you use this research, please cite:

```bibtex
@article{seal2026aadhaar,
  title={Comprehensive Statistical and Machine Learning Analysis of UIDAI Aadhaar Enrollment Data},
  author={Seal, Shuvam Banerji and Mishra, Alok and Poddar, Aheli},
  journal={[Journal Name]},
  year={2026}
}
```

### License

© 2026 Shuvam Banerji Seal, Alok Mishra, Aheli Poddar. All rights reserved.
