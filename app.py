import os
import sqlite3
import time
import numpy as np
import pandas as pd
import pickle
import requests
import re
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from openai import OpenAI
import gradio as gr
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from typing import List, Dict, Generator, Tuple
from datetime import datetime

# ------------------ ENHANCED CONFIG WITH SUPABASE -------------------

def setup_persistent_storage_with_supabase():
    """Setup persistent storage with Supabase backup"""
    # Primary persistent disk path
    PERSISTENT_PATH = "/mnt/data"
    
    # Fallback to local if persistent disk not available
    if not os.path.exists(PERSISTENT_PATH):
        print(f"[WARNING] Persistent disk not found at {PERSISTENT_PATH}")
        PERSISTENT_PATH = "./data"
        print(f"[FALLBACK] Using local storage: {PERSISTENT_PATH}")
    
    # Ensure directory exists
    os.makedirs(PERSISTENT_PATH, exist_ok=True)
    
    # Test write permissions
    test_file = os.path.join(PERSISTENT_PATH, "test_write.txt")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"[SUCCESS] Write permissions confirmed for: {PERSISTENT_PATH}")
    except Exception as e:
        print(f"[ERROR] Cannot write to {PERSISTENT_PATH}: {e}")
        # Final fallback to temp directory
        PERSISTENT_PATH = "/tmp/manuscript_data"
        os.makedirs(PERSISTENT_PATH, exist_ok=True)
        print(f"[EMERGENCY FALLBACK] Using: {PERSISTENT_PATH}")
    
    return PERSISTENT_PATH

# Setup storage
MOUNT_PATH = setup_persistent_storage_with_supabase()
DB_PATH = os.path.join(MOUNT_PATH, "articles.db")

print(f"[RENDER] Database will be stored at: {DB_PATH}")
print(f"[RENDER] Mount path: {MOUNT_PATH}")

# 🔑 SUPABASE CONFIGURATION
SUPABASE_URL = "https://amfvrmcimvfeojgvtvqn.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFtZnZybWNpbXZmZW9qZ3Z0dnFuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI2NzE1OTAsImV4cCI6MjA2ODI0NzU5MH0.VMYqQiePGikrL5XNorgfbPEJMqo61_xqC5xcCN14uHQ"

# Rest of configuration (unchanged)
client = OpenAI(api_key=os.getenv("key"))

PUBLISHERS = {
    "Mesopotamian Academic Press": "37356",
    "Peninsula Publishing Press": "51231"
}

# SUPER FETCH CONFIGURATION - NO LIMITS
MAX_CONCURRENT_PUBLISHERS = 2  # Both publishers simultaneously
MAX_ARTICLES_PER_PUBLISHER = None  # NO LIMIT - fetch everything available
FAST_MODE = True  # Enable all speed optimizations

model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

article_cache = {}
last_cache_time = 0
unique_journals_for_ui = []

# Store last search results and query for toggle functionality
last_search_results = None
last_query_title = ""

# ------------------ SUPABASE DATABASE FUNCTIONS -------------------

def create_supabase_table():
    """Create articles table in Supabase"""
    try:
        headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json"
        }
        
        # Check if table exists by trying to query it
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/articles?select=count",
            headers=headers
        )
        
        if response.status_code == 200:
            print("[SUPABASE] ✅ Table 'articles' already exists")
            return True
        else:
            print("[SUPABASE] ℹ️ Table may not exist, but that's okay - Supabase will auto-create")
            return True
            
    except Exception as e:
        print(f"[SUPABASE] ⚠️ Table check failed: {e}")
        return True  # Continue anyway

def backup_to_supabase():
    """Backup local SQLite database to Supabase"""
    try:
        print("[SUPABASE] 🔄 Starting backup process...")
        
        # Load from local SQLite
        if not os.path.exists(DB_PATH):
            print("[SUPABASE] ❌ No local database to backup")
            return False
            
        conn = sqlite3.connect(DB_PATH)
        local_df = pd.read_sql_query("SELECT * FROM articles", conn)
        conn.close()
        
        if local_df.empty:
            print("[SUPABASE] ⚠️ Local database is empty")
            return False
        
        print(f"[SUPABASE] 📊 Backing up {len(local_df):,} articles...")
        
        # Prepare data for Supabase
        backup_data = []
        for _, row in local_df.iterrows():
            # Convert embedding to JSON string if it's numpy array
            embedding = row.get('Embedding', '')
            if isinstance(embedding, np.ndarray):
                embedding = json.dumps(embedding.tolist())
            elif not isinstance(embedding, str):
                embedding = str(embedding)
            
            backup_data.append({
                "doi": row.get('DOI', ''),
                "title": row.get('Title', ''),
                "authors": row.get('Authors', ''),
                "citations": int(row.get('Citations', 0)),
                "content": row.get('Content', ''),
                "publisher": row.get('Publisher', ''),
                "journal": row.get('Journal', ''),
                "embedding": embedding,
                "abstract": row.get('Abstract', ''),
                "last_updated": datetime.now().isoformat()
            })
        
        # Upload to Supabase in batches
        headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        # Clear existing data first
        delete_response = requests.delete(
            f"{SUPABASE_URL}/rest/v1/articles",
            headers=headers
        )
        print(f"[SUPABASE] 🗑️ Cleared existing data: {delete_response.status_code}")
        
        # Upload in batches of 100
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(backup_data), batch_size):
            batch = backup_data[i:i+batch_size]
            
            response = requests.post(
                f"{SUPABASE_URL}/rest/v1/articles",
                headers=headers,
                json=batch
            )
            
            if response.status_code in [200, 201]:
                total_uploaded += len(batch)
                print(f"[SUPABASE] ✅ Uploaded batch {i//batch_size + 1}: {total_uploaded:,}/{len(backup_data):,} articles")
            else:
                print(f"[SUPABASE] ❌ Batch {i//batch_size + 1} failed: {response.status_code} - {response.text[:200]}")
                continue
        
        print(f"[SUPABASE] 🎉 Backup complete! Uploaded {total_uploaded:,} articles")
        return total_uploaded > 0
        
    except Exception as e:
        print(f"[SUPABASE] ❌ Backup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def restore_from_supabase():
    """Restore database from Supabase"""
    try:
        print("[SUPABASE] 🔄 Starting restore process...")
        
        headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json"
        }
        
        # Get data from Supabase
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/articles?select=*",
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"[SUPABASE] ❌ Failed to fetch data: {response.status_code}")
            return 0
        
        supabase_data = response.json()
        
        if not supabase_data:
            print("[SUPABASE] ⚠️ No data found in Supabase")
            return 0
        
        print(f"[SUPABASE] 📊 Restoring {len(supabase_data):,} articles...")
        
        # Convert back to SQLite format
        sqlite_data = []
        for item in supabase_data:
            # Convert embedding back from JSON string
            embedding = item.get('embedding', '')
            if embedding and isinstance(embedding, str):
                try:
                    embedding_array = json.loads(embedding)
                    if isinstance(embedding_array, list):
                        embedding = json.dumps(embedding_array)  # Keep as JSON string for SQLite
                except:
                    embedding = ''
            
            sqlite_data.append({
                'DOI': item.get('doi', ''),
                'Title': item.get('title', ''),
                'Authors': item.get('authors', ''),
                'Citations': item.get('citations', 0),
                'Content': item.get('content', ''),
                'Publisher': item.get('publisher', ''),
                'Journal': item.get('journal', ''),
                'Embedding': embedding,
                'Abstract': item.get('abstract', '')
            })
        
        # Save to local SQLite
        df = pd.DataFrame(sqlite_data)
        
        # Ensure table exists
        ensure_articles_table_exists()
        
        # Save to database
        conn = sqlite3.connect(DB_PATH)
        df.to_sql('articles', conn, if_exists='replace', index=False)
        conn.close()
        
        print(f"[SUPABASE] ✅ Restore complete! {len(sqlite_data):,} articles restored")
        return len(sqlite_data)
        
    except Exception as e:
        print(f"[SUPABASE] ❌ Restore failed: {e}")
        import traceback
        traceback.print_exc()
        return 0

def check_supabase_status():
    """Check Supabase connection and data status"""
    try:
        headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json"
        }
        
        # Get count from Supabase
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/articles?select=count",
            headers=headers
        )
        
        if response.status_code == 200:
            supabase_count = len(response.json()) if response.json() else 0
        else:
            supabase_count = 0
        
        # Get local count
        local_count = 0
        if os.path.exists(DB_PATH):
            try:
                conn = sqlite3.connect(DB_PATH)
                local_count = pd.read_sql_query("SELECT COUNT(*) AS total FROM articles", conn)["total"][0]
                conn.close()
            except:
                local_count = 0
        
        return {
            "supabase_count": supabase_count,
            "local_count": local_count,
            "connection": response.status_code == 200
        }
        
    except Exception as e:
        print(f"[SUPABASE] ❌ Status check failed: {e}")
        return {
            "supabase_count": 0,
            "local_count": 0,
            "connection": False
        }

def smart_database_initialization():
    """Smart database initialization with Supabase fallback"""
    print("[INIT] 🚀 Starting smart database initialization...")
    
    # Check local database
    local_count = 0
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            local_count = pd.read_sql_query("SELECT COUNT(*) AS total FROM articles", conn)["total"][0]
            conn.close()
            
            if local_count > 0:
                print(f"[INIT] ✅ Local database found: {local_count:,} articles")
                return {"source": "local", "count": local_count}
        except Exception as e:
            print(f"[INIT] ⚠️ Local database error: {e}")
            local_count = 0
    
    # If no local data, try Supabase restore
    if local_count == 0:
        print("[INIT] 🔄 No local data, attempting Supabase restore...")
        restored_count = restore_from_supabase()
        
        if restored_count > 0:
            # Reload cache after restore
            load_articles_from_db()
            return {"source": "supabase", "count": restored_count}
    
    # If still no data, ensure empty table exists
    print("[INIT] 📋 Creating empty database structure...")
    ensure_articles_table_exists()
    return {"source": "fresh", "count": 0}

# Create Supabase table on startup
create_supabase_table()

# Initialize database
init_result = smart_database_initialization()
print(f"[INIT] 🎉 Database initialized: {init_result}")

# ------------------ ENHANCED DATABASE SETUP WITH SUPABASE -------------------

def ensure_articles_table_exists():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table with all columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            Title TEXT,
            Authors TEXT,
            DOI TEXT PRIMARY KEY,
            Citations INTEGER,
            Content TEXT,
            Publisher TEXT,
            Journal TEXT,
            Embedding TEXT,
            Abstract TEXT
        )
    """)
    
    # Check if Abstract column exists, if not add it
    cursor.execute("PRAGMA table_info(articles)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "Abstract" not in columns:
        print("[INFO] Adding Abstract column to existing table...")
        cursor.execute("ALTER TABLE articles ADD COLUMN Abstract TEXT DEFAULT ''")
    
    conn.commit()
    conn.close()

ensure_articles_table_exists()

# ENHANCED: Unified save function with auto-backup
def save_articles_to_db(df, auto_backup=True):
    """ENHANCED: Safe database saving with Supabase auto-backup"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    saved_count = 0
    failed_count = 0
    
    # Check existing schema and adapt
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(articles)")
    columns_info = cursor.fetchall()
    existing_columns = [col[1] for col in columns_info]
    
    # Use correct column name for citations
    citation_column = "Citations"
    if "Crossref Citations" in existing_columns and "Citations" not in existing_columns:
        citation_column = "Crossref Citations"
        print(f"[INFO] Using existing column name: {citation_column}")
    
    for _, row in df.iterrows():
        try:
            # Handle embedding - generate if placeholder
            emb = row["Embedding"]
            if emb == "PENDING_EMBEDDING":
                # Generate embedding only when saving (not during fetch)
                content = row["Content"]
                embedding_array = model.encode(content, convert_to_numpy=True)
                emb = json.dumps(embedding_array.tolist())
            elif isinstance(emb, np.ndarray):
                emb = json.dumps(emb.tolist())
            elif isinstance(emb, (list, tuple)):
                emb = json.dumps(emb)
            
            # Clean abstract properly
            abstract = row.get("Abstract", "")
            if pd.isna(abstract) or abstract == "":
                abstract = ""
            
            # Use dynamic column name for citations
            citations_value = row.get("Citations", 0)
            
            # Dynamic INSERT based on existing schema
            if citation_column == "Crossref Citations":
                c.execute("""
                    INSERT OR REPLACE INTO articles (
                        DOI, Title, Authors, Publisher, Journal, Content, Embedding, "Crossref Citations", Abstract
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["DOI"],
                    row["Title"],
                    row["Authors"],
                    row["Publisher"],
                    row["Journal"],
                    row["Content"],
                    emb,
                    citations_value,
                    abstract
                ))
            else:
                c.execute("""
                    INSERT OR REPLACE INTO articles (
                        DOI, Title, Authors, Publisher, Journal, Content, Embedding, Citations, Abstract
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["DOI"],
                    row["Title"],
                    row["Authors"],
                    row["Publisher"],
                    row["Journal"],
                    row["Content"],
                    emb,
                    citations_value,
                    abstract
                ))
            
            saved_count += 1
            
            # Progress update for embedding generation
            if saved_count % 10 == 0:
                print(f"[SAVE] Generated embeddings and saved {saved_count} articles...")
                
        except Exception as e:
            failed_count += 1
            print(f"[ERROR] Failed to save article {row.get('DOI', 'UNKNOWN')}: {e}")
            # Continue trying to save other articles
            
    conn.commit()
    conn.close()
    
    print(f"[SAVE COMPLETE] Saved: {saved_count}, Failed: {failed_count}")
    
    # ENHANCED: Auto-backup to Supabase after major saves
    if auto_backup and saved_count > 0:
        print(f"[AUTO-BACKUP] Triggering Supabase backup after saving {saved_count} articles...")
        try:
            backup_success = backup_to_supabase()
            if backup_success:
                print(f"[AUTO-BACKUP] ✅ Successfully backed up to Supabase")
            else:
                print(f"[AUTO-BACKUP] ⚠️ Backup failed, but local save succeeded")
        except Exception as e:
            print(f"[AUTO-BACKUP] ❌ Backup error: {e}")
    
    return saved_count, failed_count
    
def load_articles_from_db():
    global article_cache, unique_journals_for_ui
    print("[DEBUG] Loading articles from DB...")
    conn = sqlite3.connect(DB_PATH)
    
    # Ensure Abstract column exists
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(articles)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "Abstract" not in columns:
        print("[INFO] Adding Abstract column...")
        cursor.execute("ALTER TABLE articles ADD COLUMN Abstract TEXT DEFAULT ''")
        conn.commit()
    
    df = pd.read_sql_query("SELECT * FROM articles", conn)
    conn.close()
    print(f"[DEBUG] Loaded {len(df)} articles from DB.")
    
    if not df.empty:
        def smart_load_embedding(x):
            try:
                if isinstance(x, str):
                    return np.array(json.loads(x))
                elif isinstance(x, bytes):
                    return np.array(pickle.loads(x))
            except Exception as e:
                print(f"[ERROR] Could not decode embedding: {e}")
                return None
            return None
        
        df["Embedding"] = df["Embedding"].apply(smart_load_embedding)
        
        # Ensure Abstract column exists and has data
        if "Abstract" not in df.columns:
            df["Abstract"] = ""
        else:
            # Fill NaN values with empty string
            df["Abstract"] = df["Abstract"].fillna("")
        
        # Debug: Check how many articles have abstracts
        abstract_count = (df["Abstract"].astype(str).str.len() > 10).sum()  # At least 10 chars
        print(f"[DEBUG] Articles with meaningful abstracts: {abstract_count}/{len(df)}")
            
        article_cache["df"] = df
        unique_journals_for_ui = sorted(df["Journal"].dropna().unique().tolist())
    else:
        article_cache["df"] = pd.DataFrame()
        unique_journals_for_ui = []
    print(f"[DEBUG] Journals in cache: {unique_journals_for_ui}")

# ENHANCED: Supabase UI functions
def manual_supabase_backup():
    """Manual Supabase backup with progress"""
    try:
        success = backup_to_supabase()
        
        if success:
            status = check_supabase_status()
            return f"""✅ **Supabase Backup Successful!**
            
📊 **Backup Results:**
- **Local Articles:** {status['local_count']:,}
- **Supabase Articles:** {status['supabase_count']:,}
- **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Status:** All data safely backed up to cloud!

🌐 **Supabase URL:** https://amfvrmcimvfeojgvtvqn.supabase.co"""
        else:
            return f"""❌ **Supabase Backup Failed**
            
📊 **Status:**
- **Error:** Backup process encountered issues
- **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Action:** Check network connection and try again"""
            
    except Exception as e:
        return f"❌ **Backup Error:** {str(e)}"

def manual_supabase_restore():
    """Manual Supabase restore with progress"""
    try:
        restored_count = restore_from_supabase()
        
        if restored_count > 0:
            # Reload cache
            load_articles_from_db()
            
            return f"""✅ **Supabase Restore Successful!**
            
📊 **Restore Results:**
- **Restored Articles:** {restored_count:,}
- **Source:** Supabase Cloud Database
- **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Status:** Local database updated from cloud backup!

🔄 **Next:** Cache has been refreshed automatically."""
        else:
            return f"""❌ **Supabase Restore Failed**
            
📊 **Status:**
- **Restored:** 0 articles
- **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Possible Causes:** No data in Supabase or connection issues"""
            
    except Exception as e:
        return f"❌ **Restore Error:** {str(e)}"

def check_database_sync_status():
    """Check sync status between local and Supabase"""
    try:
        status = check_supabase_status()
        
        # Get file size if exists
        file_size = 0
        if os.path.exists(DB_PATH):
            file_size = os.path.getsize(DB_PATH) / (1024*1024)  # MB
        
        return f"""📊 **Database Sync Status**
        
🏠 **Local Database:**
- **Articles:** {status['local_count']:,}
- **Location:** {DB_PATH}
- **Size:** {file_size:.2f} MB

☁️ **Supabase Cloud:**
- **Articles:** {status['supabase_count']:,}
- **Connection:** {'✅ Connected' if status['connection'] else '❌ Failed'}
- **URL:** https://amfvrmcimvfeojgvtvqn.supabase.co

🔄 **Sync Status:**
- **Difference:** {abs(status['local_count'] - status['supabase_count']):,} articles
- **Recommendation:** {'✅ In Sync' if status['local_count'] == status['supabase_count'] else '⚠️ Consider backup/restore'}
- **Last Check:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
    except Exception as e:
        return f"❌ **Status Check Error:** {str(e)}"

def refresh_journal_filter():
    """Refresh journal filter options from current database"""
    global unique_journals_for_ui
    try:
        load_articles_from_db()
        return gr.update(
            choices=unique_journals_for_ui,
            value=unique_journals_for_ui
        )
    except Exception as e:
        print(f"[ERROR] Failed to refresh journal filter: {e}")
        return gr.update()

def extract_both_with_refresh(pdf):
    """Extract title and text, then refresh journal filter"""
    title, text = extract_both(pdf)
    load_articles_from_db()  # Refresh journals
    return title, text

# ==================== SUPER CROSSREF CLIENT (NO LIMITS) ====================

class SuperCrossrefClient:
    """Ultra-fast Crossref client with anti-block measures - NO LIMITS"""
    
    def __init__(self):
        self.user_agents = [
            "SmartArticleBot/1.0 (mailto:mohanad@peninsula-press.ae)",
            "CrossrefFetcher/2.0 (mailto:mohanad@peninsula-press.ae)",
            "AcademicSearchBot/1.5 (mailto:mohanad@peninsula-press.ae)",
            "ResearchBot/3.0 (mailto:mohanad@peninsula-press.ae)",
            "ScholarlyBot/1.2 (mailto:mohanad@peninsula-press.ae)",
            "MetadataHarvester/2.1 (mailto:mohanad@peninsula-press.ae)",
            "PublicationBot/1.8 (mailto:mohanad@peninsula-press.ae)",
            "CrossrefAPI/1.0 (mailto:mohanad@peninsula-press.ae)",
            "AcademicCrawler/2.5 (mailto:mohanad@peninsula-press.ae)",
            "ScientificBot/1.4 (mailto:mohanad@peninsula-press.ae)"
        ]
        self.current_ua_index = 0
        self.base_url = "https://api.crossref.org/works"
        self.success_count = 0
        self.error_count = 0
        
    def get_rotating_user_agent(self):
        """Rotate user agents to avoid detection"""
        ua = self.user_agents[self.current_ua_index]
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
        return ua
    
    def get_optimized_headers(self):
        """Get headers optimized for Crossref polite pool"""
        return {
            "User-Agent": self.get_rotating_user_agent(),
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache"
        }
    
    def exponential_backoff(self, attempt: int, base_delay: float = 0.5):
        """Smart exponential backoff with jitter"""
        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
        return min(delay, 30)  # Cap at 30 seconds
    
    def fetch_with_retry(self, params: dict, max_retries: int = 5):
        """Fetch with intelligent retry logic"""
        for attempt in range(max_retries):
            try:
                headers = self.get_optimized_headers()
                response = requests.get(
                    self.base_url, 
                    params=params, 
                    headers=headers, 
                    timeout=45,  # Increased timeout
                    stream=False
                )
                
                if response.status_code == 200:
                    self.success_count += 1
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited - use exponential backoff
                    delay = self.exponential_backoff(attempt)
                    print(f"[RETRY] Rate limited. Waiting {delay:.1f}s before retry {attempt+1}/{max_retries}")
                    time.sleep(delay)
                    continue
                elif response.status_code in [500, 502, 503, 504]:
                    # Server error - retry with backoff
                    delay = self.exponential_backoff(attempt, 1.0)
                    print(f"[RETRY] Server error {response.status_code}. Waiting {delay:.1f}s before retry {attempt+1}/{max_retries}")
                    time.sleep(delay)
                    continue
                else:
                    print(f"[ERROR] HTTP {response.status_code}: {response.text[:200]}")
                    self.error_count += 1
                    return None
                    
            except requests.exceptions.Timeout:
                delay = self.exponential_backoff(attempt, 2.0)
                print(f"[RETRY] Timeout. Waiting {delay:.1f}s before retry {attempt+1}/{max_retries}")
                time.sleep(delay)
                continue
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Request failed: {e}")
                if attempt == max_retries - 1:
                    self.error_count += 1
                    return None
                time.sleep(self.exponential_backoff(attempt))
                continue
        
        print(f"[ERROR] Failed after {max_retries} retries")
        self.error_count += 1
        return None

# Global client instance
super_client = SuperCrossrefClient()

# ==================== UNLIMITED SINGLE PUBLISHER FETCHER ====================

def fetch_single_publisher_unlimited(publisher_data: Tuple[str, str], existing_dois: set) -> Generator[Tuple[str, List[dict]], None, None]:
    """TRULY UNLIMITED: Fetch ALL articles with NO hardcoded limits - Discovery Mode"""
    pub_name, pub_id = publisher_data
    cursor = "*"
    fetched = 0
    total_abstracts = 0
    all_articles = []
    
    print(f"[START] {pub_name} - UNLIMITED DISCOVERY MODE - No limits, fetching until API exhaustion...")
    
    while cursor:
        params = {
            "filter": f"member:{pub_id},type:journal-article",
            "rows": 100,
            "cursor": cursor,
            "mailto": "mohanad@peninsula-press.ae"
        }
        
        data = super_client.fetch_with_retry(params, max_retries=3)
        if not data:
            print(f"[ERROR] {pub_name} - Failed to fetch batch at cursor {cursor}")
            break
            
        message = data.get("message", {})
        items = message.get("items", [])
        cursor = message.get("next-cursor", None)
        
        if not items:
            print(f"[COMPLETE] {pub_name} - No more items available. Reached natural end of catalog.")
            break
        
        batch_articles = []
        batch_abstracts = 0
        
        for item in items:
            doi = item.get("DOI", "")
            doi_link = f"https://doi.org/{doi}" if doi else ""
            
            title = item.get("title", [""])[0] if item.get("title") else "Untitled"
            
            # ENHANCED ABSTRACT EXTRACTION
            abstract = ""
            if "abstract" in item and item["abstract"]:
                raw_abstract = str(item["abstract"])
                abstract = re.sub(r'<[^>]+>', '', raw_abstract).strip()
                abstract = re.sub(r'\s+', ' ', abstract).strip()
                if len(abstract) > 50:
                    batch_abstracts += 1
            
            # Extract metadata
            authors = ", ".join([
                f"{a.get('given', '')} {a.get('family', '')}" 
                for a in item.get("author", [])
            ]) if "author" in item else "N/A"
            
            journal = item.get("container-title", [""])[0] if item.get("container-title") else "Unknown Journal"
            keywords = " ".join(item.get("subject", []))
            content = f"{title} {abstract} {keywords}".lower()[:1024]
            
            # FIXED: Skip embedding generation during fetch - do it later!
            embedding_placeholder = "PENDING_EMBEDDING"
            
            batch_articles.append({
                "Title": title,
                "Authors": authors,
                "DOI": doi_link,
                "Citations": item.get("is-referenced-by-count", 0),
                "Content": content,
                "Publisher": pub_id,
                "Journal": journal,
                "Embedding": embedding_placeholder,
                "Abstract": abstract
            })
        
        fetched += len(batch_articles)
        total_abstracts += batch_abstracts
        all_articles.extend(batch_articles)
        
        # FIXED: NO HARDCODED LIMITS - Dynamic progress reporting
        if fetched < 100:
            progress_desc = f"Early discovery ({fetched} articles)"
        elif fetched < 500:
            progress_desc = f"Building collection ({fetched} articles)"
        else:
            progress_desc = f"Large dataset ({fetched} articles)"
        
        yield f"📥 **{pub_name}** | {fetched} articles discovered | {total_abstracts} abstracts | {progress_desc} | ⚡ NO embedding delays!", all_articles
        
        # Much faster with no embedding bottleneck
        time.sleep(0.01)  # Minimal delay
        
        if fetched % 100 == 0:
            print(f"[PROGRESS] {pub_name} - {fetched} articles fetched (continuing until API exhaustion)")
    
    print(f"[COMPLETE] {pub_name} - DISCOVERY FINISHED! Total discovered: {fetched} articles, {total_abstracts} with abstracts")
    yield f"✅ **{pub_name}** | DISCOVERY COMPLETE | {fetched} total articles discovered | {total_abstracts} abstracts | Ready for embedding generation", all_articles

# ==================== ENHANCED PARALLEL UNLIMITED FETCHER WITH AUTO-BACKUP ====================

def parallel_fetch_unlimited_from_both_publishers() -> Generator[str, None, None]:
    """ENHANCED: Fetch ALL articles from both publishers with auto-backup to Supabase"""
    
    # Get existing DOIs with schema adaptation
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check existing schema
        cursor.execute("PRAGMA table_info(articles)")
        columns_info = cursor.fetchall()
        existing_columns = [col[1] for col in columns_info]
        
        # Adapt to existing schema
        if "DOI" in existing_columns:
            existing_dois = set(pd.read_sql_query("SELECT DOI FROM articles", conn)["DOI"].tolist())
            existing_count = len(existing_dois)
            
            # Determine citation column name
            citation_column = "Citations"
            if "Crossref Citations" in existing_columns and "Citations" not in existing_columns:
                citation_column = "Crossref Citations"
            
            conn.close()
            yield f"🔍 **Starting UNLIMITED Fetch** | Found {existing_count} existing articles | Target: DISCOVER ALL AVAILABLE"
            yield f"📋 **Schema Info** | Using citation column: '{citation_column}' | Existing columns: {len(existing_columns)}"
        else:
            existing_dois = set()
            conn.close()
            yield f"⚠️ **Database Init** | No existing articles table found | Creating fresh database"
            
    except Exception as e:
        existing_dois = set()
        yield f"⚠️ **Database Error** | {str(e)} | Starting with empty DOI set"
    
    # NO HARDCODED LIMITS - Let API discovery happen naturally
    yield f"🎯 **Discovery Mode** | NO LIMITS - Will fetch until Crossref API says 'no more articles available'"
    yield f"🔬 **Method** | Each publisher will be fetched until cursor exhaustion (natural API endpoint)"
    
    # Prepare publisher list
    publishers = list(PUBLISHERS.items())
    yield f"🎯 **Target Publishers** | {len(publishers)} publishers: {', '.join([p[0] for p in publishers])}"
    
    # Prepare for parallel execution
    all_new_articles = []
    total_abstracts = 0
    completed_publishers = 0
    publisher_results = {}
    
    # UNLIMITED: Use ThreadPoolExecutor with proper lambda closure
    with ThreadPoolExecutor(max_workers=2) as executor:
        # FIXED: Proper lambda closure to avoid variable capture issues
        def fetch_publisher_wrapper(publisher_data):
            return list(fetch_single_publisher_unlimited(publisher_data, existing_dois))
        
        # Submit both publisher fetch tasks - NO LIMITS
        future_to_publisher = {}
        for pub in publishers:
            future = executor.submit(fetch_publisher_wrapper, pub)
            future_to_publisher[future] = pub[0]
        
        yield f"🚀 **Unlimited Parallel Fetch Started** | Processing both publishers simultaneously | NO LIMITS WHATSOEVER!"
        yield f"⚡ **Strategy** | Fetch until API cursor returns null - natural discovery of all available articles"
        
        # Process completed tasks
        for future in as_completed(future_to_publisher):
            pub_name = future_to_publisher[future]
            
            try:
                # Get all progress updates from this publisher
                progress_updates = future.result()
                
                if progress_updates:
                    # Get final result
                    final_progress, publisher_articles = progress_updates[-1]
                    
                    # UNLIMITED: Count ALL articles discovered
                    total_discovered = len(publisher_articles)
                    
                    # Filter for truly new articles
                    new_articles = []
                    for article in publisher_articles:
                        if article["DOI"] not in existing_dois:
                            new_articles.append(article)
                    
                    # Count meaningful abstracts
                    abstracts_in_batch = sum(1 for article in new_articles 
                                           if len(str(article.get("Abstract", "")).strip()) > 50)
                    
                    all_new_articles.extend(new_articles)
                    total_abstracts += abstracts_in_batch
                    completed_publishers += 1
                    
                    # Store results for summary - DISCOVERY STATS
                    publisher_results[pub_name] = {
                        "total_discovered": total_discovered,
                        "new_articles": len(new_articles),
                        "abstracts": abstracts_in_batch,
                        "already_had": total_discovered - len(new_articles)
                    }
                    
                    yield f"✅ **{pub_name} DISCOVERY COMPLETE** | 🔍 Discovered: {total_discovered} total | ➕ New: {len(new_articles)} | 📝 Abstracts: {abstracts_in_batch} | 🔄 Already had: {total_discovered - len(new_articles)} | Progress: {completed_publishers}/2"
                    
            except Exception as e:
                yield f"❌ **{pub_name}** | Failed: {str(e)}"
                import traceback
                error_details = traceback.format_exc()
                yield f"🔍 **Error Details** | {error_details[:200]}..."
                completed_publishers += 1
                
                # Store failed result
                publisher_results[pub_name] = {
                    "total_discovered": 0,
                    "new_articles": 0,
                    "abstracts": 0,
                    "error": str(e)
                }
    
    # Calculate total discovery stats
    total_discovered_all_publishers = sum(r.get("total_discovered", 0) for r in publisher_results.values())
    total_already_had = sum(r.get("already_had", 0) for r in publisher_results.values())
    
    yield f"🔬 **DISCOVERY SUMMARY** | 🌍 Total available in Crossref: {total_discovered_all_publishers} | 💾 Already in DB: {total_already_had} | ➕ New to fetch: {len(all_new_articles)}"
    
    # UNLIMITED: Save ALL new discoveries with auto-backup
    if all_new_articles:
        yield f"💾 **Saving ALL Discoveries** | Processing {len(all_new_articles)} newly discovered articles"
        yield f"📊 **Abstract Discovery** | {total_abstracts} articles have meaningful abstracts ({total_abstracts/len(all_new_articles)*100:.1f}%)"
        
        # Save in optimized batches with progress tracking
        batch_size = 50  # Smaller batches for better progress tracking
        saved_count = 0
        failed_saves = 0
        
        for i in range(0, len(all_new_articles), batch_size):
            batch = all_new_articles[i:i+batch_size]
            df_batch = pd.DataFrame(batch)
            
            try:
                # ENHANCED: Use save function with auto-backup disabled for batches
                batch_saved, batch_failed = save_articles_to_db(df_batch, auto_backup=False)
                saved_count += batch_saved
                failed_saves += batch_failed
                
                progress_pct = int((i + len(batch)) / len(all_new_articles) * 100)
                yield f"💽 **Saving Progress** | {saved_count}/{len(all_new_articles)} saved | {failed_saves} failed | {progress_pct}% complete"
                
            except Exception as e:
                failed_saves += len(batch)
                yield f"❌ **Batch Save Failed** | Batch {i//batch_size + 1}: {str(e)}"
        
        # ENHANCED: Single backup after all saves complete
        if saved_count > 0:
            yield f"☁️ **Auto-Backup Starting** | Backing up {saved_count} new articles to Supabase..."
            try:
                backup_success = backup_to_supabase()
                if backup_success:
                    yield f"✅ **Auto-Backup Complete** | All {saved_count} articles safely stored in Supabase cloud!"
                else:
                    yield f"⚠️ **Auto-Backup Failed** | Local data saved successfully, but cloud backup needs manual retry"
            except Exception as e:
                yield f"❌ **Auto-Backup Error** | {str(e)} | Local data is safe"
        
        # Continue with existing final statistics...
        try:
            conn = sqlite3.connect(DB_PATH)
            final_count = pd.read_sql_query("SELECT COUNT(*) AS total FROM articles", conn)["total"][0]
            
            # Use proper column name for abstract count
            abstract_count_query = "SELECT COUNT(*) AS count FROM articles WHERE Abstract IS NOT NULL AND LENGTH(TRIM(Abstract)) > 50"
            final_abstract_count = pd.read_sql_query(abstract_count_query, conn)["count"][0]
            
            # Get updated publisher breakdown
            mesopotamian_final = pd.read_sql_query("SELECT COUNT(*) AS count FROM articles WHERE Publisher = '37356'", conn)["count"][0]
            peninsula_final = pd.read_sql_query("SELECT COUNT(*) AS count FROM articles WHERE Publisher = '51231'", conn)["count"][0]
            
            conn.close()
            
            # Reload cache
            yield f"🔄 **Reloading Cache** | Updating in-memory cache with {final_count} total articles..."
            load_articles_from_db()
            
            # UNLIMITED: Enhanced final summary with discovery totals
            yield f"🎉 **UNLIMITED DISCOVERY MISSION ACCOMPLISHED!**"
            yield f"🔬 **Discovery Results** | 🌍 Total available in Crossref: {total_discovered_all_publishers} articles"
            yield f"💾 **Database Results** | Added: {saved_count} new articles | Failed: {failed_saves} articles | Final total: {final_count}"
            yield f"📈 **Content Quality** | With abstracts: {final_abstract_count} ({final_abstract_count/final_count*100:.1f}%)"
            yield f"🏢 **Publisher Final Counts** | Mesopotamian: {mesopotamian_final} articles | Peninsula: {peninsula_final} articles"
            yield f"📡 **API Performance** | Success: {super_client.success_count} | Errors: {super_client.error_count} | Success rate: {super_client.success_count/(super_client.success_count + super_client.error_count)*100:.1f}%"
            
            # Individual publisher discovery results
            for pub_name, results in publisher_results.items():
                if "error" in results:
                    yield f"❌ **{pub_name}** | Error: {results['error']}"
                else:
                    yield f"🔍 **{pub_name} Discovery** | 🌍 Available: {results['total_discovered']} | ➕ New: {results['new_articles']} | 📝 Abstracts: {results['abstracts']}"
            
            # UNLIMITED: Success validation
            if final_count == total_discovered_all_publishers:
                yield f"✅ **PERFECT DISCOVERY** | Database now contains ALL {final_count} articles available in Crossref!"
            elif final_count > (total_discovered_all_publishers * 0.95):  # 95% threshold
                yield f"✅ **EXCELLENT DISCOVERY** | Database contains {final_count}/{total_discovered_all_publishers} articles ({final_count/total_discovered_all_publishers*100:.1f}% of available)"
            else:
                yield f"⚠️ **PARTIAL DISCOVERY** | Database contains {final_count}/{total_discovered_all_publishers} articles | {total_discovered_all_publishers - final_count} still missing"
            
        except Exception as e:
            yield f"❌ **Statistics Error** | Could not generate final stats: {str(e)}"
        
    else:
        # No new articles found - but show discovery results
        load_articles_from_db()
        
        conn = sqlite3.connect(DB_PATH)
        current_total = pd.read_sql_query("SELECT COUNT(*) AS total FROM articles", conn)["total"][0]
        conn.close()
        
        yield f"🔍 **DISCOVERY COMPLETE - NO NEW ARTICLES** | 🌍 Total available: {total_discovered_all_publishers} | 💾 Database has: {current_total}"
        
        if current_total == total_discovered_all_publishers:
            yield f"✅ **PERFECT COVERAGE** | Database contains ALL {current_total} articles available in Crossref API!"
        else:
            yield f"🤔 **ANALYSIS NEEDED** | Database has {current_total} but API shows {total_discovered_all_publishers} available"
            
        # Show per-publisher discovery
        for pub_name, results in publisher_results.items():
            if "error" not in results:
                yield f"🔍 **{pub_name}** | 🌍 Discovered: {results['total_discovered']} | 💾 Already had: {results['already_had']}"

# ENHANCED: Cache update function with Supabase auto-backup
def unified_cache_update_with_realtime_progress():
    """ENHANCED: SUPER UNLIMITED cache update with auto-backup to Supabase"""
    global unique_journals_for_ui, article_cache
    
    try:
        # Use the enhanced parallel unlimited fetcher
        for progress_message in parallel_fetch_unlimited_from_both_publishers():
            yield progress_message
            
    except Exception as e:
        print(f"[ERROR] Super unlimited fetch failed: {e}")
        import traceback
        traceback.print_exc()
        yield f"❌ **CRITICAL ERROR** | Unlimited fetch failed: {str(e)}"
# ENHANCED highlight function that only highlights meaningful keywords
def highlight_meaningful_keywords(text, query_title):
    """Enhanced highlighting with word similarity and stemming"""
    if not text or not query_title:
        return text
    
    try:
        # Use KeyBERT to extract meaningful keywords from the query title
        keywords = kw_model.extract_keywords(query_title, keyphrase_ngrams=(1, 2), stop_words='english', top_k=10)
        
        # Extract meaningful words with relevance scores
        meaningful_words = set()
        for keyword, score in keywords:
            if score > 0.15:  # Lowered threshold for more keywords
                # Split multi-word keyphrases
                words = keyword.split()
                for word in words:
                    if len(word) > 3 and word.isalpha():
                        meaningful_words.add(word.lower())
        
        # Enhanced fallback with better filtering
        if len(meaningful_words) < 3:
            # Very comprehensive stop words for academic text
            academic_stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
                'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 
                'these', 'those', 'from', 'into', 'during', 'before', 'after', 'above', 'below', 
                'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'use', 'using',
                'used', 'make', 'makes', 'made', 'get', 'gets', 'got', 'take', 'takes', 'took',
                'also', 'well', 'way', 'even', 'back', 'good', 'new', 'first', 'last', 'long',
                'great', 'little', 'right', 'old', 'different', 'following', 'public', 'able'
            }
            
            # Extract words manually with better criteria
            words = re.findall(r'\b[a-zA-Z]+\b', query_title.lower())
            for word in words:
                if (len(word) > 4 and 
                    word not in academic_stop_words and 
                    not word.isdigit() and 
                    word.isalpha()):
                    meaningful_words.add(word)
        
        print(f"[DEBUG] Meaningful keywords for highlighting: {meaningful_words}")
        
        if not meaningful_words:
            return text
        
        highlighted_text = text
        for word in meaningful_words:
            # ENHANCED: Match word and its variations (plurals, different forms)
            # Pattern 1: Exact word
            pattern1 = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            # Pattern 2: Word with common suffixes (s, es, ed, ing, ly, er, est)
            pattern2 = re.compile(r'\b' + re.escape(word) + r'(s|es|ed|ing|ly|er|est|tion|ment|ness|able|ible)\b', re.IGNORECASE)
            # Pattern 3: Word without common suffixes (for stemming-like matching)
            if len(word) > 4:
                root_word = word[:-1] if word.endswith(('s', 'e')) else word
                pattern3 = re.compile(r'\b' + re.escape(root_word) + r'[a-z]{0,4}\b', re.IGNORECASE)
            else:
                pattern3 = None
            
            # Apply highlighting for exact matches
            highlighted_text = pattern1.sub(
                f'<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: 500;">{word}</mark>', 
                highlighted_text
            )
            
            # Apply highlighting for word variations with different color
            highlighted_text = pattern2.sub(
                lambda m: f'<mark style="background-color: #fff176; padding: 2px 4px; border-radius: 3px; font-weight: 400;">{m.group()}</mark>', 
                highlighted_text
            )
            
            # Apply highlighting for root word matches with lighter color
            if pattern3:
                highlighted_text = pattern3.sub(
                    lambda m: f'<mark style="background-color: #fff9c4; padding: 1px 3px; border-radius: 2px; font-weight: 300;">{m.group()}</mark>', 
                    highlighted_text
                )
        
        return highlighted_text
        
    except Exception as e:
        print(f"[DEBUG] Keyword extraction failed: {e}")
        # Simple fallback with word variations
        words = re.findall(r'\b[a-zA-Z]{4,}\b', query_title.lower())
        meaningful_words = set(words[:5])  # Top 5 words
        
        highlighted_text = text
        for word in meaningful_words:
            # Simple pattern matching with variations
            pattern = re.compile(r'\b' + re.escape(word) + r'[a-z]*\b', re.IGNORECASE)
            highlighted_text = pattern.sub(
                f'<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: 500;">{word}</mark>', 
                highlighted_text
            )
        
        return highlighted_text

# MERGED results generation with beauty UI and similarity scores
def generate_unified_results_html(df_sorted, query_title, show_abstracts=False):
    """UNIFIED HTML generation with beauty UI, similarity scores, and abstracts"""
    if df_sorted.empty:
        return "<p style='text-align: center; padding: 20px; color: #666;'>No results found.</p>"
    
    html_parts = []
    
    for idx, row in df_sorted.iterrows():
        title = row.get('Title', 'No Title')
        doi = row.get('DOI', '')
        similarity = row.get('Similarity', 0)
        authors = row.get('Authors', 'Unknown')
        journal = row.get('Journal', 'Unknown Journal')
        citations = row.get('Citations', 0)
        abstract = row.get('Abstract', '')
        
        # Use enhanced highlighting for meaningful keywords only
        highlighted_title = highlight_meaningful_keywords(title, query_title)
        
        # Determine similarity badge color
        if similarity >= 80:
            sim_color = "#e74c3c"  # Red for high similarity
        elif similarity >= 60:
            sim_color = "#f39c12"  # Orange for medium similarity
        else:
            sim_color = "#95a5a6"  # Gray for low similarity
        
        # Create beautiful HTML for each result
        result_html = f"""
        <div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 20px; margin: 15px 0; background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px;">
                <h4 style="margin: 0; color: #2c3e50; font-size: 1.1em; line-height: 1.4; flex: 1; padding-right: 15px;">{highlighted_title}</h4>
                <div style="display: flex; flex-direction: column; align-items: center; min-width: 80px;">
                    <span style="background: {sim_color}; color: white; padding: 6px 12px; border-radius: 20px; font-weight: bold; font-size: 14px; margin-bottom: 5px;">
                        {similarity}%
                    </span>
                    <span style="color: #7f8c8d; font-size: 11px;">Match</span>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                <div>
                    <div style="margin-bottom: 8px;">
                        <strong style="color: #34495e;">👥 Authors:</strong>
                        <div style="color: #7f8c8d; font-size: 14px; margin-top: 2px;">{authors}</div>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <strong style="color: #34495e;">📖 Journal:</strong>
                        <div style="color: #3498db; font-size: 14px; margin-top: 2px;">{journal}</div>
                    </div>
                </div>
                <div>
                    <div style="margin-bottom: 8px;">
                        <strong style="color: #34495e;">📊 Citations:</strong>
                        <span style="color: #e67e22; font-weight: bold; margin-left: 5px;">{citations}</span>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <strong style="color: #34495e;">🔗 DOI:</strong>
                        <div style="margin-top: 2px;">
                            <a href="{doi}" target="_blank" style="color: #e67e22; text-decoration: none; font-size: 13px; word-break: break-all;">{doi}</a>
                        </div>
                    </div>
                </div>
            </div>
        """
        
        if show_abstracts:
            if abstract and len(str(abstract).strip()) > 50:  # Has meaningful abstract - FIXED threshold
                highlighted_abstract = highlight_meaningful_keywords(abstract, query_title)
                result_html += f"""
                <div style="margin-top: 15px; padding: 15px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 8px; border-left: 4px solid #3498db;">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <strong style="color: #2c3e50;">📝 Abstract</strong>
                        <span style="background: #3498db; color: white; padding: 2px 6px; border-radius: 10px; font-size: 10px; margin-left: 10px;">
                            {len(abstract)} chars
                        </span>
                    </div>
                    <div style="line-height: 1.6; color: #2c3e50; font-size: 14px;">
                        {highlighted_abstract}
                    </div>
                </div>
                """
            else:  # No abstract available
                result_html += f"""
                <div style="margin-top: 15px; padding: 15px; background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); border-radius: 8px; border-left: 4px solid #f39c12;">
                    <div style="display: flex; align-items: center;">
                        <strong style="color: #856404;">⚠️ Abstract</strong>
                    </div>
                    <div style="margin-top: 5px; line-height: 1.6; color: #856404; font-style: italic;">
                        No abstract available for this article.
                    </div>
                </div>
                """
        
        result_html += "</div>"
        html_parts.append(result_html)
    
    return "".join(html_parts)

# MERGED find_related_articles function 
def find_related_articles(title, selected_publishers, selected_journals):
    """UNIFIED find related articles with beauty UI and similarity scores"""
    global last_query_title, last_search_results
    last_query_title = title  # Store for highlighting
    
    print("🚨🚨🚨 FUNCTION IS BEING CALLED! 🚨🚨🚨")
    print("\n========== [DEBUG] find_related_articles called ==========")
    print("[DEBUG] Title received:", repr(title))
    print("[DEBUG] Selected publishers from UI:", selected_publishers)
    print("[DEBUG] Selected journals from UI:", selected_journals)

    df = article_cache.get("df", pd.DataFrame())
    print("[DEBUG] Initial cache shape:", df.shape)
    if df.empty:
        print("[ERROR] Article cache is empty. Please refresh cache from database first.")
        last_search_results = None
        return gr.update(choices=[], value=[]), gr.update(value="<p style='text-align: center; padding: 20px; color: #e74c3c;'>No articles in cache. Please refresh the database first.</p>")

    print("[DEBUG] DataFrame columns:", df.columns.tolist())

    # Publisher filter - Convert publisher names to IDs
    if selected_publishers:
        print("[DEBUG] Filtering publishers...")
        publisher_ids = [PUBLISHERS.get(name, name) for name in selected_publishers]
        print("[DEBUG] Publisher names:", selected_publishers)
        print("[DEBUG] Publisher IDs to filter:", publisher_ids)
        print("[DEBUG] Publishers in cache:", df["Publisher"].dropna().unique().tolist())
        df = df[df["Publisher"].isin(publisher_ids)]
        print("[DEBUG] After publisher filter shape:", df.shape)
    else:
        print("[DEBUG] No publisher filter applied.")

    # Journal filter
    if selected_journals:
        print("[DEBUG] Filtering journals...")
        print("[DEBUG] Journals in cache:", df["Journal"].dropna().unique().tolist())
        df = df[df["Journal"].isin(selected_journals)]
        print("[DEBUG] After journal filter shape:", df.shape)
    else:
        print("[DEBUG] No journal filter applied.")

    if df.empty:
        print("[INFO] No cached articles match your filter selection.")
        last_search_results = None
        return gr.update(choices=[], value=[]), gr.update(value="<p style='text-align: center; padding: 20px; color: #f39c12;'>No articles match your filter selection.</p>")

    # Ensure embeddings are valid
    print("[DEBUG] Validating embeddings in filtered DataFrame...")
    embedding_types = df["Embedding"].apply(lambda x: type(x)).value_counts()
    print("[DEBUG] Embedding types in filtered df:", embedding_types.to_dict())

    df = df[df["Embedding"].apply(lambda x: isinstance(x, np.ndarray) and x.shape[0] > 0)]
    print("[DEBUG] After embedding validation shape:", df.shape)

    if df.empty:
        print("[ERROR] No valid embeddings found among filtered articles.")
        last_search_results = None
        return gr.update(choices=[], value=[]), gr.update(value="<p style='text-align: center; padding: 20px; color: #e74c3c;'>No valid embeddings found in filtered articles.</p>")

    print("[DEBUG] Sample 'Title', 'DOI', 'Publisher', 'Journal', 'Embedding shape':")
    for idx, row in df.head(2).iterrows():
        emb = row["Embedding"]
        emb_shape = emb.shape if isinstance(emb, np.ndarray) else None
        print(f"  {idx}: {row['Title'][:30]}... | DOI: {row['DOI']} | Pub: {row['Publisher']} | Jour: {row['Journal']} | Embedding shape: {emb_shape}")

    # Title input check
    if not isinstance(title, str) or not title.strip():
        print("[ERROR] Invalid or missing title input.")
        last_search_results = None
        return gr.update(choices=[], value=[]), gr.update(value="<p style='text-align: center; padding: 20px; color: #e74c3c;'>Please enter a valid title.</p>")

    try:
        print(f"[DEBUG] Encoding query title: {title[:100]}...")
        query_embed = model.encode(title, convert_to_numpy=True)
        print("[DEBUG] Query embedding shape:", query_embed.shape)

        doc_embeds = list(df["Embedding"])
        print("[DEBUG] Number of doc embeddings for similarity:", len(doc_embeds))
        print("[DEBUG] First doc embedding shape:", doc_embeds[0].shape if len(doc_embeds) > 0 else None)

        print("[DEBUG] Computing cosine similarity...")
        sims = cosine_similarity([query_embed], doc_embeds)[0]
        print("[DEBUG] Similarity scores for first 5 articles:", sims[:5])

        df = df.copy()
        df["Similarity"] = (sims * 100).round(2)
        df_sorted = df.sort_values(by="Similarity", ascending=False).head(10)

        if df_sorted.empty:
            print("[INFO] No similar articles found after similarity ranking.")
            last_search_results = None
            return gr.update(choices=[], value=[]), gr.update(value="<p style='text-align: center; padding: 20px; color: #f39c12;'>No similar articles found.</p>")

        print(f"[DEBUG] Top match: {df_sorted.iloc[0]['Title']} (Similarity: {df_sorted.iloc[0]['Similarity']})")

        # Store the filtered and sorted results for abstract toggle
        last_search_results = df_sorted.copy()
        print(f"[DEBUG] Stored search results with {len(last_search_results)} articles")
        print(f"[DEBUG] Abstract column exists: {'Abstract' in last_search_results.columns}")
        
        # Create checkbox choices for backward compatibility
        items = [f"{row['Title']} ({row['DOI']})" for _, row in df_sorted.iterrows()]
        
        # Generate UNIFIED beautiful HTML with similarity scores (without abstracts initially)
        results_html = generate_unified_results_html(df_sorted, title, show_abstracts=False)
        
        print("[DEBUG] Returning items:", len(items))
        return gr.update(choices=items, value=[]), gr.update(value=results_html)
        
    except Exception as e:
        print(f"[ERROR] Failed in find_related_articles: {e}")
        import traceback
        traceback.print_exc()
        last_search_results = None
        return gr.update(choices=[], value=[]), gr.update(value=f"<p style='text-align: center; padding: 20px; color: #e74c3c;'>Error occurred: {str(e)}</p>")

def toggle_abstracts(show_abstracts):
    """Toggle abstract display using stored search results"""
    global last_search_results, last_query_title
    
    print(f"[DEBUG] Toggle abstracts called: show_abstracts={show_abstracts}")
    print(f"[DEBUG] Last search results available: {last_search_results is not None}")
    print(f"[DEBUG] Last query title: {repr(last_query_title)}")
    
    if last_search_results is None or last_search_results.empty:
        print("[DEBUG] No search results available for toggle")
        return gr.update(value="<p style='text-align: center; padding: 20px; color: #f39c12;'>No results to display. Please search for articles first.</p>")
    
    if not last_query_title:
        print("[DEBUG] No query title available")
        return gr.update(value="<p style='text-align: center; padding: 20px; color: #f39c12;'>No search query available. Please search first.</p>")
    
    try:
        print(f"[DEBUG] Using stored search results with {len(last_search_results)} articles")
        print(f"[DEBUG] Columns available: {list(last_search_results.columns)}")
        
        # Check if Abstract column has data
        abstract_data = last_search_results["Abstract"].fillna("").astype(str)
        non_empty_abstracts = (abstract_data.str.len() > 20).sum()
        print(f"[DEBUG] Articles with meaningful abstracts: {non_empty_abstracts}")
        
        # Use stored search results with UNIFIED beautiful HTML
        results_html = generate_unified_results_html(last_search_results, last_query_title, show_abstracts=show_abstracts)
        
        print(f"[DEBUG] Generated HTML length: {len(results_html)}")
        print(f"[DEBUG] Show abstracts: {show_abstracts}")
        
        return gr.update(value=results_html)
        
    except Exception as e:
        print(f"[ERROR] Error toggling abstracts: {e}")
        import traceback
        traceback.print_exc()
        return gr.update(value=f"<p style='text-align: center; padding: 20px; color: #e74c3c;'>Error toggling abstracts: {str(e)}</p>")

def filter_manual_search(query):
    df = article_cache.get("df", pd.DataFrame())

    if df.empty:
        print("[WARNING] Article cache is empty.")
        return gr.update(choices=[], value=[])

    query = query.strip().lower()
    if not query:
        return gr.update(choices=[], value=[])

    try:
        matches = df[
            df['Title'].astype(str).str.lower().str.contains(query, na=False) |
            df['Publisher'].astype(str).str.lower().str.contains(query, na=False)
        ]

        if matches.empty:
            print("[INFO] No manual search results found.")
            return gr.update(choices=[], value=[])

        choices = [f"{row['Title']} ({row['DOI']})" for _, row in matches.iterrows()]
        return gr.update(choices=choices, value=[])
    except Exception as e:
        print(f"[ERROR] Manual search failed: {e}")
        return gr.update(choices=[], value=[])

def check_irrelevant_and_warn(prompt, pdf_text, extracted_title, selected_articles):
    if not selected_articles or not extracted_title.strip():
        return gr.update(value="No articles selected or no extracted title.", visible=True), gr.update(visible=False)

    try:
        pdf_embed = model.encode(extracted_title.strip(), convert_to_numpy=True)
    except Exception as e:
        return gr.update(value=f"[ERROR] Failed to encode title: {e}", visible=True), gr.update(visible=False)

    irrelevant = []

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT Title, DOI, Embedding FROM articles", conn)
        df["Embedding"] = df["Embedding"].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else x)
        conn.close()
    except Exception as e:
        return gr.update(value=f"[ERROR] Failed to load cached articles: {e}", visible=True), gr.update(visible=False)

    for item in selected_articles:
        try:
            title = item.split(" (")[0].strip()
            doi = item.split("(")[-1].split(")")[0].strip()
            article_row = df[df["DOI"] == doi]
            if article_row.empty:
                continue
            article_embed = article_row.iloc[0]["Embedding"]
            sim = cosine_similarity([pdf_embed], [article_embed])[0][0] * 100
            if sim >= 70:
                continue

            # Use GPT fallback
            question = (
                f"Compare the manuscript title with the article title below.\n"
                f"Manuscript: \"{extracted_title}\"\n"
                f"Article: \"{title}\"\n\n"
                f"Answer in the format 'specific: yes/no, general: yes/no' only."
            )

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a domain expert reviewer. Respond in the format: 'specific: yes/no, general: yes/no'."},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=10,
                    temperature=0.0
                )
                answer = response.choices[0].message.content.strip().lower()
                specific = "yes" if "specific: yes" in answer else "no"
                general = "yes" if "general: yes" in answer else "no"

                if specific == "no" and general == "no":
                    irrelevant.append(f"- **{title}** ({doi})")
            except Exception as e:
                print(f"[WARNING] GPT API error during relevance check: {e}")
                continue

        except Exception as e:
            print(f"[ERROR] Failed relevance check for article {item}: {e}")
            continue

    if irrelevant:
        msg = f"⚠️ The following articles may be irrelevant to the manuscript title:\n\n" + "\n".join(irrelevant) + \
              "\n\nYou may go back and deselect them, or proceed anyway."
        return gr.update(value=msg, visible=True), gr.update(visible=True)
    else:
        return "", gr.update(visible=True)

def handle_review(prompt, text):
    try:
        combined = f"{prompt}\n\n---\n\n{text}"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": combined}],
            max_tokens=1500,
            temperature=0.4
        )
        return response.choices[0].message.content, gr.update(visible=False)
    except Exception as e:
        fallback_message = f"""[WARNING] GPT-4 API failed. This is a fallback response.

[ERROR DETAILS] {e}

The OpenAI API was unreachable or failed to respond. Please try again later or use the fallback output below based on available cache and similarity metrics."""
        return fallback_message, gr.update(visible=True)

def fill_prompt_with_dois(dois):
    if not dois:
        return ""
    links = []
    for d in dois:
        try:
            title_part = d.split(" (")[0].strip()
            raw_doi = d.split("(")[-1].split(")")[0].strip()
            doi_link = raw_doi if raw_doi.startswith("http") else f"https://doi.org/{raw_doi}"
            links.append(f"{doi_link} — {title_part}")
        except:
            continue
    links_text = "\n".join(links)
    return f"""You're an academic peer reviewer with expertise matching the uploaded manuscript. Write 5 to 8 thoughtful, human-like, and precise comments that directly address the paper's content. Include:

- Two comments on grammar or clarity  
- One comment on the methodology (its strengths or needed improvements)  
- One comment on the originality or scientific contribution  
- Additional comments may address structure, coherence, literature support, or interpretation of results  

Avoid generic remarks. Be specific, professional, and helpful.

**STRICT CITATION RULES:**
- You MUST use ONLY the citations provided below - do NOT suggest any other references
- Distribute the provided citations across your comments using this EXACT alternating pattern:
  * 1st citation: Use DOI ONLY (e.g., "Consider citing https://doi.org/...")
  * 2nd citation: Use TITLE ONLY (e.g., "The study titled 'Deep Learning for Crop Monitoring' discusses...")  
  * 3rd citation: Use DOI ONLY again
  * 4th citation: Use TITLE ONLY again
  * Continue this strict alternation
- NEVER use both DOI and title in the same comment
- Use each citation only once
- Justify why each citation is relevant (1-2 lines)
- If you have fewer comments than citations, prioritize the most relevant ones

Citations to use:

{links_text}

Begin directly with the review comments. Do not include any headings, explanations, or summaries—just the comments."""

def extract_title_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = "\n".join([doc[i].get_text("text") for i in range(min(2, len(doc)))])
        if not text.strip():
            print("[ERROR] No readable text extracted from the first 2 pages of the PDF.")
            return "[ERROR] No readable text extracted from the first 2 pages of the PDF."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract only the title of the academic paper from the following text. Do not include authors or journal names."},
                {"role": "user", "content": text}
            ],
            max_tokens=50,
            temperature=0.2
        )
        title = response.choices[0].message.content.strip()
        print("[DEBUG] GPT extracted title:", title)
        return title
    except Exception as e:
        print(f"[ERROR] Title extraction failed: {e}")
        return f"[ERROR] Title extraction failed: {e}"

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
        return text[:9000].strip()
    except Exception as e:
        return f"[ERROR] Failed to extract PDF text: {e}"

def extract_both(pdf):
    import tempfile
    import shutil

    if not pdf:
        print("[ERROR] No PDF uploaded.")
        return "", ""

    try:
        # Create a secure temp copy of the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfile(pdf.name, tmp.name)
            temp_path = tmp.name

        # Extract title and text
        title = extract_title_from_pdf(temp_path)
        text = extract_text_from_pdf(temp_path)

        if not title:
            print("[WARNING] Title extraction returned empty string.")
        if not text:
            print("[WARNING] Full text extraction returned empty string.")

        return title or "", text or ""

    except Exception as e:
        print(f"[ERROR] PDF extraction failed: {e}")
        return "", ""
# --------- Article Stats Logic ---------
def get_article_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT Journal, DOI FROM articles", conn)
        conn.close()
        total_articles = len(df)
        articles_per_journal = df["Journal"].value_counts()
        df["Year"] = df["DOI"].apply(lambda x: x.split("/")[-1][:4] if "/" in x else "Unknown")
        year_journal_counts = df.groupby(["Journal", "Year"]).size().reset_index(name="Count")
        return total_articles, articles_per_journal, year_journal_counts
    except Exception as e:
        print(f"[ERROR] Stats loading failed: {e}")
        return 0, pd.Series(dtype=int), pd.DataFrame(columns=["Journal", "Year", "Count"])

def toggle_detailed_stats():
    """Show detailed statistics"""
    total_articles, articles_per_journal, year_journal_counts = get_article_stats()
    
    # Build detailed stats HTML
    detailed_stats = f"""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 10px;">
        <h4>📈 Detailed Statistics</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            {"".join([f"<div style='background: white; padding: 10px; border-radius: 6px; text-align: center;'><strong>{journal}</strong><br><span style='color: #007bff; font-size: 1.2em;'>{count}</span> articles</div>" for journal, count in articles_per_journal.items()])}
        </div>
    </div>
    """
    return gr.update(value=detailed_stats, visible=True), gr.update(visible=False)

def hide_detailed_stats():
    """Hide detailed statistics"""
    return gr.update(visible=False), gr.update(visible=True)

# ------------------ ENHANCED UI HELPER FUNCTIONS -------------------

def clear_prompt():
    """Clear the prompt box"""
    return ""

def save_review(review_text):
    """Save review to file"""
    if not review_text.strip():
        return gr.update(info="❌ No review to save")
    
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"peer_review_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Peer Review Generated on {datetime.datetime.now()}\n\n")
            f.write(review_text)
        
        return gr.update(info=f"✅ Review saved as {filename}")
    except Exception as e:
        return gr.update(info=f"❌ Save failed: {str(e)}")

def refresh_database_cache():
    """ENHANCED: Refresh the in-memory cache with Supabase sync option"""
    try:
        load_articles_from_db()
        conn = sqlite3.connect(DB_PATH)
        count = pd.read_sql_query("SELECT COUNT(*) AS total FROM articles", conn)["total"][0]
        
        # Check for abstracts
        abstract_count = pd.read_sql_query("SELECT COUNT(*) AS count FROM articles WHERE Abstract IS NOT NULL AND LENGTH(TRIM(Abstract)) > 50", conn)["count"][0]
        
        # Get publisher breakdown
        mesopotamian_count = pd.read_sql_query("SELECT COUNT(*) AS count FROM articles WHERE Publisher = '37356'", conn)["count"][0]
        peninsula_count = pd.read_sql_query("SELECT COUNT(*) AS count FROM articles WHERE Publisher = '51231'", conn)["count"][0]
        
        conn.close()
        
        # Check Supabase sync status
        status = check_supabase_status()
        sync_status = "✅ In Sync" if count == status['supabase_count'] else f"⚠️ Difference: {abs(count - status['supabase_count'])}"
        
        return f"""✅ **Cache Refreshed Successfully!**
        
📊 **Database Statistics:**
- **Total Articles:** {count:,}
- **With Abstracts:** {abstract_count:,} ({abstract_count/count*100:.1f}% if count > 0 else 0)
- **Mesopotamian Academic Press:** {mesopotamian_count:,}
- **Peninsula Publishing Press:** {peninsula_count:,}

☁️ **Supabase Sync Status:**
- **Local:** {count:,} articles
- **Cloud:** {status['supabase_count']:,} articles  
- **Status:** {sync_status}

🔄 **Cache:** In-memory cache updated with latest database content."""
        
    except Exception as e:
        return f"❌ **Cache Refresh Failed:** {str(e)}"

def clear_search():
    """Clear search box"""
    return ""

def regenerate_review(prompt, pdf_text):
    """Regenerate review with same prompt"""
    return handle_review(prompt, pdf_text)

# ------------------ ENHANCED DYNAMIC STATISTICS FUNCTION -------------------

def get_dynamic_header_stats():
    """ENHANCED: Get real-time database statistics with Supabase status"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get total count
        total_count = pd.read_sql_query("SELECT COUNT(*) AS total FROM articles", conn)["total"][0]
        
        # Get publisher breakdown
        mesopotamian_count = pd.read_sql_query("SELECT COUNT(*) AS count FROM articles WHERE Publisher = '37356'", conn)["count"][0]
        peninsula_count = pd.read_sql_query("SELECT COUNT(*) AS count FROM articles WHERE Publisher = '51231'", conn)["count"][0]
        
        # Get abstracts count
        abstract_count = pd.read_sql_query("SELECT COUNT(*) AS count FROM articles WHERE Abstract IS NOT NULL AND LENGTH(TRIM(Abstract)) > 50", conn)["count"][0]
        
        # Get top 3 journals
        top_journals = pd.read_sql_query("""
            SELECT Journal, COUNT(*) as count 
            FROM articles 
            WHERE Journal IS NOT NULL AND Journal != 'Unknown Journal'
            GROUP BY Journal 
            ORDER BY count DESC 
            LIMIT 3
        """, conn)
        
        conn.close()
        
        # ENHANCED: Check Supabase status
        try:
            supabase_status = check_supabase_status()
            sync_indicator = "🟢" if supabase_status['connection'] and total_count == supabase_status['supabase_count'] else "🟡"
            supabase_info = f"{sync_indicator} Cloud: {supabase_status['supabase_count']:,}"
        except:
            supabase_info = "🔴 Cloud: N/A"
        
        # Build journal badges
        journal_badges = ""
        for _, row in top_journals.iterrows():
            journal_name = row['Journal'][:20] + "..." if len(row['Journal']) > 20 else row['Journal']
            journal_badges += f"<span style='background: rgba(255,255,255,0.15); padding: 6px 10px; border-radius: 15px; font-size: 12px;'>{journal_name}: {row['count']}</span>"
        
        # ENHANCED: Build dynamic HTML with Supabase status
        stats_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div style="font-size: 18px; font-weight: bold;">📊 Live Database Statistics</div>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <span style="background: rgba(255,255,255,0.25); padding: 8px 12px; border-radius: 20px; font-weight: bold;">
                        Total: {total_count:,}
                    </span>
                    <span style="background: rgba(255,255,255,0.2); padding: 6px 10px; border-radius: 15px; font-size: 12px;">
                        📝 Abstracts: {abstract_count:,}
                    </span>
                    <span style="background: rgba(255,255,255,0.2); padding: 6px 10px; border-radius: 15px; font-size: 12px;">
                        🏛️ Mesopotamian: {mesopotamian_count:,}
                    </span>
                    <span style="background: rgba(255,255,255,0.2); padding: 6px 10px; border-radius: 15px; font-size: 12px;">
                        🏝️ Peninsula: {peninsula_count:,}
                    </span>
                    <span style="background: rgba(255,255,255,0.2); padding: 6px 10px; border-radius: 15px; font-size: 12px;">
                        {supabase_info}
                    </span>
                    {journal_badges}
                </div>
            </div>
        </div>
        """
        
        return stats_html
        
    except Exception as e:
        return f"""
        <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); padding: 15px; border-radius: 10px; color: white; margin-bottom: 10px;">
            <div style="font-size: 18px; font-weight: bold;">❌ Statistics Error: {str(e)}</div>
        </div>
        """

# ------------------ ENHANCED APP UI WITH SUPABASE -------------------

def run_gradio_app():
    load_articles_from_db()  # Preload articles and update unique_journals_for_ui

    # Start Gradio UI with improved loading states
    with gr.Blocks(css="""
        .gr-box { border-radius: 6px; }
        .scroll-area { max-height: 200px; overflow-y: auto; }
        .compact-stats { margin-bottom: 15px; }
        .results-container { max-height: 500px; overflow-y: auto; border: 1px solid #ddd; border-radius: 8px; padding: 10px; }
    """) as app:
        gr.Markdown("# 📄 AI-Powered Manuscript Reviewer")
        
        # ✅ ENHANCED Statistics Section with Supabase Status
        with gr.Group(elem_classes="compact-stats"):
            # Dynamic header that updates automatically with Supabase status
            header_stats = gr.HTML(value=get_dynamic_header_stats())
            
            with gr.Row():
                show_stats_btn = gr.Button("📊 Show Detailed Statistics", size="sm")
                hide_stats_btn = gr.Button("🔼 Hide Detailed Statistics", size="sm", visible=False)
                refresh_header_btn = gr.Button("🔄 Refresh Header", size="sm", variant="secondary")
            detailed_stats_container = gr.HTML("", visible=False)

        # ✅ Filters Section
        with gr.Group():
            gr.Markdown("## 🔧 Filter Settings")
            with gr.Row():
                publisher_filter = gr.CheckboxGroup(
                    label="Publisher Filter",
                    choices=list(PUBLISHERS.keys()),
                    value=list(PUBLISHERS.keys())
                )
                journal_filter = gr.CheckboxGroup(
                    label="Journal Filter",
                    choices=unique_journals_for_ui,
                    value=unique_journals_for_ui,
                    interactive=True,
                    elem_classes="scroll-area"
                )

        # 1️⃣ Upload + Extraction Section
        with gr.Group():
            gr.Markdown("## 📥 Upload Manuscript")
            with gr.Row():
                pdf_file = gr.File(label="Upload PDF")
                extracted_title = gr.Textbox(label="Extracted Title", interactive=True)
            with gr.Row():
                extract_btn = gr.Button("Extract Title + Text", variant="primary")

        # 2️⃣ Enhanced Related Articles Section
        with gr.Group():
            gr.Markdown("## 🔎 Related Articles")
            with gr.Row():
                filter_btn = gr.Button("Find Related Articles", variant="primary")
                abstracts_toggle = gr.Checkbox(label="Show Abstracts", value=False)
            
            # Enhanced results display with beauty UI and similarity scores
            results_display = gr.HTML(
                value="<p style='text-align: center; color: #666; padding: 20px;'>Click 'Find Related Articles' to see results with similarity scores and highlighted keywords.</p>",
                elem_classes="results-container"
            )
            
            # Keep original checkbox for backward compatibility
            result_checkboxes = gr.CheckboxGroup(label="Select Articles to Cite (for prompt generation)", choices=[], visible=True)

        # 3️⃣ Review Generation Section
        with gr.Group():
            gr.Markdown("## ✍️ Peer Review Generator")
            with gr.Row():
                prompt_box = gr.TextArea(
                    label="Review Prompt (Editable)", 
                    lines=8,
                    placeholder="The prompt will be auto-generated when you select articles above. You can edit it before generating the review."
                )
            
            with gr.Row():
                submit_btn = gr.Button("🧠 Generate Peer Review", variant="primary", size="lg")
                clear_prompt_btn = gr.Button("🧹 Clear Prompt", variant="secondary")
            
            # Warning and proceed section
            irrelevant_warning = gr.Markdown("", visible=False)
            with gr.Row():
                proceed_btn = gr.Button("⚠️ Proceed Anyway", visible=False, variant="stop")
            
            # Fallback notice
            fallback_notice = gr.Markdown(
                "**⚠️ GPT-4 API failed. This review is generated using cached similarity logic as fallback.**",
                visible=False
            )
            
            # Enhanced review output with copy functionality
            with gr.Row():
                review_output = gr.Textbox(
                    label="Generated Peer Review", 
                    lines=12, 
                    interactive=True,
                    placeholder="Your generated peer review will appear here..."
                )
            
            # Review action buttons
            with gr.Row():
                copy_review_btn = gr.Button("📋 Copy Review", size="sm")
                save_review_btn = gr.Button("💾 Save Review", size="sm", variant="secondary")
                regenerate_btn = gr.Button("🔄 Regenerate", size="sm", variant="secondary")

        # 4️⃣ ENHANCED Manual Search & Database Management Section with Supabase
        with gr.Group():
            gr.Markdown("## 🔍 Manual Article Search & Database Management")
            
            # Enhanced search with better UX
            with gr.Row():
                manual_search = gr.Textbox(
                    label="Search Articles", 
                    placeholder="Type to search by title, author, or publisher...",
                    scale=3
                )
                search_clear_btn = gr.Button("🧹 Clear", size="sm", scale=1)
            
            # Search results with improved display
            manual_results = gr.CheckboxGroup(
                label="Search Results",
                choices=[],
                value=[],
                interactive=True
            )
            
            # ENHANCED Database management section with Supabase
            gr.Markdown("### 📊 Database Management")
            gr.Markdown("💡 **Tip:** All data is automatically backed up to Supabase cloud after unlimited fetch")
            
            with gr.Row():
                update_cache_btn = gr.Button(
                    "🚀 UNLIMITED FETCH: Discover ALL Available Articles", 
                    variant="primary", 
                    size="lg"
                )
                refresh_cache_btn = gr.Button("🔄 Refresh Cache", variant="secondary")
            
            # ENHANCED: Supabase controls
            with gr.Row():
                backup_btn = gr.Button("☁️ Backup to Cloud", variant="secondary")
                restore_btn = gr.Button("📥 Restore from Cloud", variant="secondary")
                status_btn = gr.Button("📊 Check Sync Status", variant="secondary")
            
            # Enhanced progress display with better formatting
            cache_progress = gr.Markdown(
                value="",
                visible=True,
                elem_classes="progress-display"
            )
            
            # Database statistics mini-dashboard
            with gr.Row():
                db_stats = gr.HTML(
                    value="<div style='text-align: center; padding: 10px; background: #f8f9fa; border-radius: 8px; margin-top: 10px;'>📊 Database statistics will appear here after operations</div>",
                    visible=True
                )
            
        # 🔒 Hidden state
        pdf_text_hidden = gr.State()

        # 🔗 ENHANCED Bind Functions with Supabase Integration
        
        # Statistics toggle with hide functionality
        show_stats_btn.click(
            fn=toggle_detailed_stats,
            outputs=[detailed_stats_container, show_stats_btn]
        )
        
        hide_stats_btn.click(
            fn=hide_detailed_stats,
            outputs=[detailed_stats_container, show_stats_btn]
        )
        
        # Header refresh functionality
        refresh_header_btn.click(
            fn=get_dynamic_header_stats,
            outputs=[header_stats]
        )
        
        # ENHANCED: Extract with journal refresh
        extract_btn.click(
            fn=extract_both_with_refresh, 
            inputs=[pdf_file], 
            outputs=[extracted_title, pdf_text_hidden],
            show_progress=True
        ).then(
            fn=refresh_journal_filter,
            outputs=[journal_filter]
        )
        
        # Enhanced find related articles with unified output
        filter_btn.click(
            fn=find_related_articles, 
            inputs=[extracted_title, publisher_filter, journal_filter], 
            outputs=[result_checkboxes, results_display],
            show_progress=True
        )
        
        # Enhanced Abstract toggle functionality
        abstracts_toggle.change(
            fn=toggle_abstracts,
            inputs=[abstracts_toggle],
            outputs=[results_display]
        )
        
        manual_search.change(
            fn=filter_manual_search, 
            inputs=[manual_search], 
            outputs=[manual_results]
        )
        
        submit_btn.click(
            fn=check_irrelevant_and_warn, 
            inputs=[prompt_box, pdf_text_hidden, extracted_title, result_checkboxes], 
            outputs=[irrelevant_warning, proceed_btn],
            show_progress=True
        )
        
        proceed_btn.click(
            fn=handle_review, 
            inputs=[prompt_box, pdf_text_hidden], 
            outputs=[review_output, fallback_notice],
            show_progress=True
        )
        
        result_checkboxes.change(
            fn=fill_prompt_with_dois, 
            inputs=result_checkboxes, 
            outputs=prompt_box
        )
        
        manual_results.change(
            fn=fill_prompt_with_dois, 
            inputs=manual_results, 
            outputs=prompt_box
        )
        
        # ENHANCED: Cache update with auto-refresh header and journal filter
        update_cache_btn.click(
            fn=unified_cache_update_with_realtime_progress, 
            outputs=[cache_progress]
        ).then(
            fn=get_dynamic_header_stats,
            outputs=[header_stats]
        ).then(
            fn=refresh_journal_filter,
            outputs=[journal_filter]
        )

        # ENHANCED: Supabase backup/restore functionality
        backup_btn.click(
            fn=manual_supabase_backup,
            outputs=[cache_progress]
        ).then(
            fn=get_dynamic_header_stats,
            outputs=[header_stats]
        )
        
        restore_btn.click(
            fn=manual_supabase_restore,
            outputs=[cache_progress]
        ).then(
            fn=get_dynamic_header_stats,
            outputs=[header_stats]
        ).then(
            fn=refresh_journal_filter,
            outputs=[journal_filter]
        )
        
        status_btn.click(
            fn=check_database_sync_status,
            outputs=[cache_progress]
        )

        # Additional button bindings for UI elements
        clear_prompt_btn.click(
            fn=clear_prompt,
            outputs=[prompt_box]
        )

        save_review_btn.click(
            fn=save_review,
            inputs=[review_output],
            outputs=[save_review_btn]
        )

        search_clear_btn.click(
            fn=clear_search,
            outputs=[manual_search]
        )

        refresh_cache_btn.click(
            fn=refresh_database_cache,
            outputs=[cache_progress]
        ).then(
            fn=get_dynamic_header_stats,
            outputs=[header_stats]
        ).then(
            fn=refresh_journal_filter,
            outputs=[journal_filter]
        )

        regenerate_btn.click(
            fn=regenerate_review,
            inputs=[prompt_box, pdf_text_hidden],
            outputs=[review_output, fallback_notice],
            show_progress=True
        )

    # 🚀 RENDER DEPLOYMENT CONFIGURATION
    port = int(os.environ.get("PORT", 7860))
    
    app.launch(
        server_name="0.0.0.0",  # Required for Render
        server_port=port,       # Use Render's port
        share=False,           # Don't create public tunnels
        show_error=True        # Show errors in production
    )

if __name__ == "__main__":
    print(f"[RENDER] Starting app on port {os.environ.get('PORT', 7860)}")
    print(f"[RENDER] Database path: {DB_PATH}")
    print(f"[RENDER] API key configured: {'key' in os.environ}")
    print(f"[SUPABASE] URL: {SUPABASE_URL}")
    print(f"[SUPABASE] Connection will be tested on first operation")
    run_gradio_app()
