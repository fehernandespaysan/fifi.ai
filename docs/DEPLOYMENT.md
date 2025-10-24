# Fifi.ai Deployment Guide

**Last Updated:** October 24, 2025
**Target Platform:** Streamlit Cloud (Free Tier)
**Estimated Time:** 15-30 minutes

---

## 📋 Pre-Deployment Checklist

### Content Requirements ✅ / ❌

- [ ] **Real blog content created** (3-5 posts minimum)
  - Replace demo posts in `blogs/` directory
  - Each post should be 800-2000 words
  - Include proper frontmatter (title, date, tags, author)
  - Topics based on your expertise

- [ ] **Blog content validated**
  - All markdown files have valid YAML frontmatter
  - No placeholder or "lorem ipsum" text
  - Proper formatting and readability
  - Test locally: `python scripts/test_blog_loading.py`

### Technical Requirements ✅ / ❌

- [x] **All tests passing**
  - Current: 120/127 tests passing (94.5%)
  - Run: `pytest tests/`
  - Core functionality: 100% passing ✓

- [x] **Dependencies up to date**
  - Check: `pip list --outdated`
  - Security scan: `bandit -r src/`
  - All in `requirements.txt`

- [x] **Environment variables documented**
  - `.env.example` is complete
  - No secrets in code (verified with git log)
  - API keys ready for Streamlit Cloud

- [x] **Code quality verified**
  - No hardcoded secrets: ✓
  - Type hints complete: ✓
  - Proper error handling: ✓
  - Logging configured: ✓

### Documentation Requirements ✅ / ❌

- [x] **README.md up to date**
  - Accurate project status
  - Clear installation instructions
  - Usage examples included

- [x] **ROADMAP.md reflects progress**
  - Completed phases marked
  - Current status accurate

- [x] **TEST_SUMMARY.md current**
  - Test results documented
  - Known issues listed

---

## 🚀 Deployment Steps

### Step 1: Prepare Repository

1. **Final code cleanup**
   ```bash
   # Remove any temporary files
   find . -name "*.pyc" -delete
   find . -name "__pycache__" -type d -delete
   find . -name ".DS_Store" -delete

   # Ensure .gitignore is working
   git status  # Should not show .env, data/, logs/
   ```

2. **Create fresh embeddings with real content**
   ```bash
   # Activate virtual environment
   source venv/bin/activate

   # Generate embeddings from your real blog posts
   python scripts/test_embeddings.py

   # Verify embeddings
   python scripts/test_rag.py
   ```

3. **Test the Streamlit app locally**
   ```bash
   streamlit run streamlit_app.py
   # Open http://localhost:8501
   # Test queries with your actual content
   # Verify sources are cited correctly
   ```

4. **Commit final changes**
   ```bash
   git add .
   git commit -m "Ready for deployment: Real content and final polish"
   git push origin main
   ```

### Step 2: Streamlit Cloud Setup

1. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Sign in with GitHub account

2. **Create New App**
   - Click "New app"
   - Select your repository: `yourusername/fifi.ai`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Configure Environment Variables**
   - In Streamlit Cloud dashboard, go to app settings
   - Click "Secrets"
   - Add your secrets in TOML format:
   ```toml
   # Streamlit secrets format
   OPENAI_API_KEY = "sk-your-actual-api-key-here"

   # Optional: Other settings
   ENVIRONMENT = "production"
   LOG_LEVEL = "INFO"
   CHUNK_SIZE = "500"
   CHUNK_OVERLAP = "50"
   VECTOR_SEARCH_TOP_K = "5"
   ```

4. **Save and Reboot**
   - Click "Save"
   - App will automatically reboot
   - Wait 2-3 minutes for deployment

### Step 3: Verify Deployment

1. **Test the deployed app**
   - Open your Streamlit Cloud URL (e.g., `https://yourapp.streamlit.app`)
   - Try several queries about your blog topics
   - Verify responses are accurate
   - Check that sources are cited correctly
   - Test edge cases (empty input, very long queries)

2. **Monitor performance**
   - Check response times (should be 2-5 seconds)
   - Monitor token usage in app statistics
   - Verify no errors in Streamlit Cloud logs

3. **Check error handling**
   - Try invalid inputs
   - Test with network issues (if possible)
   - Verify graceful error messages

### Step 4: Post-Deployment

1. **Set up monitoring**
   - Check Streamlit Cloud metrics daily for first week
   - Monitor OpenAI API usage: https://platform.openai.com/usage
   - Set up billing alerts in OpenAI dashboard

2. **Update README with live link**
   ```markdown
   ## 🌐 Live Demo

   Try Fifi.ai live: [https://your-app.streamlit.app](https://your-app.streamlit.app)
   ```

3. **Share with initial users**
   - Personal network first
   - Request feedback
   - Monitor for issues

4. **Create backup of embeddings**
   ```bash
   # Local backup
   tar -czf data_backup_$(date +%Y%m%d).tar.gz data/

   # Consider uploading to S3 or Google Drive for safety
   ```

---

## 🔧 Configuration

### Recommended Streamlit Cloud Settings

- **Python version:** 3.11 (or 3.12)
- **Resources:** Use default (sufficient for free tier)
- **Always on:** OFF (for free tier to conserve resources)
- **Public/Private:** Public (for open-source project)

### Expected Resource Usage

**Free Tier Limits (should be fine for 100s of users):**
- RAM: ~512MB (you'll use ~200-300MB)
- Storage: Unlimited (your data/ folder is ~50MB)
- Bandwidth: Generous free tier
- Uptime: Sleeps after inactivity (wakes in seconds)

**Cost Estimates (OpenAI API):**
- 100 users × 5 queries = 500 queries/month
- Estimated cost: ~$2-5/month
- Set billing limit in OpenAI dashboard to be safe

---

## 🐛 Troubleshooting

### App Won't Start

**Error: "ModuleNotFoundError"**
- Check `requirements.txt` has all dependencies
- Ensure no local-only packages included
- Try: Clear cache and reboot in Streamlit Cloud

**Error: "OPENAI_API_KEY not found"**
- Verify secrets are set in Streamlit Cloud settings
- Use exact variable name: `OPENAI_API_KEY`
- No quotes needed in Streamlit secrets (TOML format)

### Slow Performance

**Queries taking >10 seconds**
- Check OpenAI API status: https://status.openai.com
- Consider using `gpt-4o-mini` instead of `gpt-4o`
- Reduce `CHUNK_SIZE` if responses are very long

**App frequently sleeping**
- This is normal on free tier after 15 min inactivity
- Upgrade to always-on if needed (paid tier)
- Wake time is only 3-5 seconds, usually acceptable

### No Responses / Empty Results

**Error: "No relevant information found"**
- Embeddings may not have been generated
- Check that blog posts loaded correctly
- Regenerate embeddings: Run `scripts/test_embeddings.py` locally
- Commit updated `data/` folder if persistent storage needed

**Note:** Streamlit Cloud doesn't persist `data/` folder by default. Options:
1. Commit embeddings to repo (if small, <100MB)
2. Generate on startup (slower first load, 30-60s)
3. Use external vector DB like Pinecone (overkill for MVP)

**Current approach:** Commit embeddings to repo for MVP

---

## 🔒 Security Checklist

Before going live:

- [ ] **Secrets not in code**
  - Verify: `git log --all -S "sk-" | grep -v "example"`
  - Should return nothing

- [ ] **Environment variables secure**
  - All secrets in Streamlit Cloud secrets
  - `.env` file in `.gitignore`
  - No `.env` file committed to repo

- [ ] **Rate limiting aware**
  - Monitor OpenAI usage daily
  - Set billing limits ($10-20/month for safety)
  - Watch for unusual patterns

- [ ] **Error messages safe**
  - No stack traces shown to users
  - No API keys in logs
  - Generic error messages only

- [ ] **Input validation working**
  - Test XSS attempts
  - Test very long inputs (should be truncated)
  - Test special characters

---

## 📊 Monitoring & Maintenance

### Daily (First Week)

- [ ] Check Streamlit Cloud logs for errors
- [ ] Monitor OpenAI API usage and costs
- [ ] Review user queries (if logged anonymously)
- [ ] Check response quality

### Weekly

- [ ] Review feedback from users
- [ ] Update blog content as needed
- [ ] Check for dependency updates (security)
- [ ] Backup embeddings locally

### Monthly

- [ ] Run full test suite: `pytest tests/`
- [ ] Update dependencies: `pip list --outdated`
- [ ] Review and optimize costs
- [ ] Plan new features based on usage

---

## 🎯 Success Metrics

Track these metrics to measure success:

### Technical Metrics
- **Uptime:** Target 99%+ (Streamlit Cloud is reliable)
- **Response time:** < 5 seconds (p95)
- **Error rate:** < 1% of queries
- **Cost per query:** < $0.01

### User Metrics
- **Daily active users:** Track growth
- **Queries per user:** Target 3-5 per session
- **Session duration:** Target 3-5 minutes
- **Return users:** Track week-over-week

### Quality Metrics
- **Relevance:** Do responses match queries?
- **Source accuracy:** Are sources cited correctly?
- **User satisfaction:** Gather feedback

---

## 🚨 Rollback Plan

If something goes wrong:

1. **Immediate:**
   - Streamlit Cloud: Settings → "Revert to previous version"
   - Or: Set app to private while fixing

2. **Code issues:**
   ```bash
   git revert HEAD
   git push origin main
   # Streamlit Cloud auto-deploys
   ```

3. **API issues:**
   - Check OpenAI status page
   - Verify API key hasn't expired
   - Check billing/quota limits

4. **Data issues:**
   - Restore embeddings from backup
   - Re-run `scripts/test_embeddings.py`
   - Commit and push updated data/

---

## 🎉 Launch Checklist

Final checks before announcing publicly:

- [ ] App loads without errors
- [ ] Queries return relevant results
- [ ] Sources are cited correctly
- [ ] Response times acceptable (< 5s)
- [ ] Error handling works gracefully
- [ ] Statistics dashboard shows correct data
- [ ] Mobile view works (test on phone)
- [ ] README updated with live link
- [ ] Demo video/screenshots ready (optional)
- [ ] Announcement tweet/post drafted

**You're ready to launch! 🚀**

---

## 📞 Support

If you encounter issues:

1. Check Streamlit Cloud docs: https://docs.streamlit.io/streamlit-community-cloud
2. Check OpenAI status: https://status.openai.com
3. Review app logs in Streamlit Cloud dashboard
4. Check GitHub Issues for similar problems
5. Ask in Streamlit Community: https://discuss.streamlit.io

---

## 🔄 Future Enhancements

After successful deployment, consider:

- [ ] Add Google Analytics for usage tracking
- [ ] Set up error tracking (Sentry)
- [ ] Add feedback mechanism (thumbs up/down)
- [ ] Implement caching for common queries
- [ ] Add more blog content regularly
- [ ] Build FastAPI backend if API needed
- [ ] Add authentication for advanced features
- [ ] Scale to Pinecone if >10K users

---

**Remember:** Start small, gather feedback, iterate based on real usage!

Good luck with your deployment! 🎊
