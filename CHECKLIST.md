# 📋 Pre-Publication Checklist

Use this checklist to ensure everything is ready before publishing to GitHub and submitting your paper revision.

## ✅ Code Preparation

### Test Functionality

- [ ] Installed all dependencies (`pip install -r requirements.txt`)
- [ ] Ran `python main.py` successfully
- [ ] Verified results generated in `results/` folder
- [ ] Verified figures generated in `figures/` folder
- [ ] Checked console output for errors
- [ ] Tested on clean Python environment

### Code Quality

- [ ] All placeholder text replaced with actual information
- [ ] No debugging print statements left in code
- [ ] No hardcoded file paths
- [ ] Code follows consistent style
- [ ] All imports work correctly
- [ ] No unused files or code

## ✅ Documentation Updates

### Personal Information

- [ ] **README.md**: Updated author name, email, institution
- [ ] **CITATION.cff**: Updated author details, ORCID (if available)
- [ ] **LICENSE**: Updated copyright holder name and year
- [ ] **main.py**: Updated contact information
- [ ] **src/**init**.py**: Updated author and email
- [ ] **QUICK_START.md**: Updated contact information
- [ ] **SETUP_GUIDE.md**: Updated contact information
- [ ] **PUBLISHING_GUIDE.md**: Updated with your details

### Repository Information

After creating GitHub repo, update:

- [ ] **README.md**: Replace `YOUR_USERNAME` with actual username
- [ ] **CITATION.cff**: Update repository URL
- [ ] **main.py**: Update GitHub URL
- [ ] **QUICK_START.md**: Update GitHub URL
- [ ] **All documentation**: Replace placeholder URLs

## ✅ Data and Files

### Required Files Present

- [ ] All CSV files in `data/` folder (4 files)
- [ ] All Python modules in `src/` folder (7 files)
- [ ] Main execution script (`main.py`)
- [ ] Requirements file (`requirements.txt`)
- [ ] README file (`README.md`)
- [ ] License file (`LICENSE`)
- [ ] Gitignore file (`.gitignore`)

### Data Verification

- [ ] CSV files are not corrupted
- [ ] File names match those in `config.py`
- [ ] Data loads correctly
- [ ] No sensitive or private data included
- [ ] Data source is cited

## ✅ GitHub Preparation

### Repository Setup

- [ ] Created GitHub account
- [ ] Decided on repository name (recommended: `mental-health-prediction`)
- [ ] Decided on visibility (should be Public for open source)
- [ ] Prepared repository description

### Repository Content

- [ ] All necessary files ready to commit
- [ ] No large unnecessary files (models, checkpoints)
- [ ] `.gitignore` properly configured
- [ ] No sensitive information (API keys, passwords)
- [ ] License file included

## ✅ Paper Updates

### Manuscript Changes

- [ ] Updated Conclusion section with GitHub URL
- [ ] Added "Code Availability" section
- [ ] Updated Abstract (optional but recommended)
- [ ] Added repository URL to relevant table/figure captions
- [ ] Verified all GitHub URLs are consistent

### Code Availability Section Includes

- [ ] Repository URL
- [ ] Brief description of contents
- [ ] Installation instructions
- [ ] Reproduction instructions
- [ ] Expected runtime
- [ ] System requirements
- [ ] License information
- [ ] Contact information

### References

- [ ] Added software citation in references
- [ ] Used proper citation format
- [ ] Included version number
- [ ] Included DOI (if using Zenodo)

## ✅ Reviewer Response

### Response Letter

- [ ] Acknowledged reviewer's concern
- [ ] Explained actions taken
- [ ] Listed manuscript changes with page numbers
- [ ] Described repository contents
- [ ] Included repository URL
- [ ] Professional and courteous tone

### Changes Document

- [ ] Created tracked changes version
- [ ] Highlighted all modifications
- [ ] Added marginal notes for major changes
- [ ] Included in submission materials

## ✅ Final Verification

### Code Testing

- [ ] Tested on fresh virtual environment
- [ ] Verified on different computer (if possible)
- [ ] Checked all dependencies install correctly
- [ ] Runtime is reasonable (5-10 minutes)
- [ ] Results match those in paper

### GitHub Repository

- [ ] Repository is public
- [ ] README displays correctly on main page
- [ ] All files are visible
- [ ] Can be cloned without authentication
- [ ] Links in README work
- [ ] Issues tab enabled (for questions)

### Documentation Quality

- [ ] README is clear and comprehensive
- [ ] Installation instructions tested
- [ ] Quick start guide is accurate
- [ ] No broken links
- [ ] No typos in critical sections
- [ ] Code examples work

## ✅ Optional Enhancements

### Nice to Have

- [ ] Added repository badges (Python version, license)
- [ ] Created example outputs folder
- [ ] Added troubleshooting section to README
- [ ] Created GitHub Issues templates
- [ ] Added Contributing guidelines
- [ ] Set up continuous integration (GitHub Actions)

### Long-term

- [ ] Registered with Zenodo for DOI
- [ ] Created release (v1.0.0)
- [ ] Tagged with topic keywords on GitHub
- [ ] Shared on social media/research networks
- [ ] Added to awesome-lists (if relevant)

## ✅ Submission Checklist

### Before Submitting Paper

- [ ] Code published to GitHub ✓
- [ ] Repository URL verified working ✓
- [ ] All paper sections updated ✓
- [ ] Response letter written ✓
- [ ] All files double-checked ✓

### Submission Materials

- [ ] Revised manuscript
- [ ] Response to reviewers letter
- [ ] Tracked changes version
- [ ] Cover letter mentioning code availability
- [ ] Any required supplementary materials

## 🎯 Quick Pre-Flight Check

**Can someone else reproduce your results?**

Test with a colleague or friend:

1. Give them only the GitHub URL
2. Ask them to follow README instructions
3. Verify they can:
   - [ ] Install dependencies
   - [ ] Run the code
   - [ ] Get results
   - [ ] Understand what it does

If yes to all → You're ready! 🚀

## 📝 Final Notes

### Timing

- [ ] Allow 1-2 hours for complete preparation
- [ ] Test during normal working hours (in case of issues)
- [ ] Don't rush - better to be thorough

### Backup

- [ ] Keep local backup of all files
- [ ] Save original notebook version
- [ ] Document any major changes

### After Publication

- [ ] Monitor GitHub issues/questions
- [ ] Be responsive to community
- [ ] Update if bugs found
- [ ] Thank contributors

## ✨ You're Ready When...

✅ Code runs without errors
✅ Documentation is complete
✅ All personal info updated
✅ Repository is public on GitHub
✅ Paper references GitHub correctly
✅ Response letter is written
✅ You're confident someone else can reproduce your work

## 🎉 Congratulations!

Once all items are checked, you're ready to:

1. Submit your paper revision
2. Share your open-source contribution
3. Join the reproducible research community

---

**Last updated**: December 2024
**Questions?** Refer to PUBLISHING_GUIDE.md or open a GitHub issue
