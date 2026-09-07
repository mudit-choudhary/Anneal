# import sqlite3
# from typing import Literal, Optional
# from pathlib import Path

# Status = Literal["downloaded", "parsed", "embedded", "error"]

# class FileRegistry:
#     def __init__(self, db_path: str = "rag_registry.db"):
#         self.db_path = db_path
#         self._init_db()
    
#     def _init_db(self):
#         with sqlite3.connect(self.db_path) as conn:
#             conn.execute("""
#                 CREATE TABLE IF NOT EXISTS file_status_table (
#                     filename TEXT PRIMARY KEY,
#                     status TEXT CHECK(status IN ('downloaded', 'parsed', 'processed', 'embedded', 'error')),
#                     error_count INTEGER DEFAULT 0,
#                     last_error TEXT,
#                     downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                     parsed_at TIMESTAMP,
#                     embedded_at TIMESTAMP
#                 );
#             """)
    
#     def update_status(self, filename: str, status: Status, error_msg: Optional[str] = None):
#         try:
#             with sqlite3.connect(self.db_path) as conn:
#                 if status == "error":
#                     conn.execute("""
#                         INSERT OR REPLACE INTO file_status_table 
#                         (filename, status, error_count, last_error, parsed_at, embedded_at) 
#                         VALUES (?, ?, 
#                             COALESCE((SELECT error_count + 1 FROM file_status_table WHERE filename = ?), 1),
#                             ?, NULL, NULL)
#                     """, (filename, status, filename, error_msg))
#                 elif status == "downloaded":
#                     conn.execute("INSERT OR IGNORE INTO file_status_table (filename, status, error_count) VALUES (?, ?, 0)", 
#                                 (filename, status))
#                 elif status == "parsed":
#                     conn.execute("UPDATE file_status_table SET status = ?, parsed_at = CURRENT_TIMESTAMP, error_count = 0 WHERE filename = ?", 
#                                 (status, filename))
#                 elif status == "embedded":
#                     conn.execute("UPDATE file_status_table SET status = ?, embedded_at = CURRENT_TIMESTAMP, error_count = 0 WHERE filename = ?", 
#                                 (status, filename))
#                 conn.commit()
#             return True
#         except Exception as e:

#             return False

#     def get_error_files(self, max_errors: int = 3) -> list[tuple[str, int, str]]:
#         """Return files with high error counts"""
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.execute("""
#                 SELECT filename, error_count, last_error 
#                 FROM file_status_table 
#                 WHERE error_count >= ? AND status = 'error'
#             """, (max_errors,))
#             return cursor.fetchall()

    
#     def get_status(self, filename: str) -> str | None:
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.execute(
#                 "SELECT status FROM file_status_table WHERE filename = ?", 
#                 (filename,)
#             )
#             result = cursor.fetchone()  # ✅ Call ONCE
#             return result[0] if result else None  # ✅ Safe indexing
    
#     def pending_files(self, target_status: Status) -> list[str]:
#         next_statuses = {"downloaded": "parsed", "parsed": "embedded"}
#         current = next_statuses.get(target_status, target_status)
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.execute("SELECT filename FROM file_status_table WHERE status = ?", (current,))
#             return [row[0] for row in cursor.fetchall()]

#     # def exists(self, filename: str) -> bool:
#     #     with sqlite3.connect(self.db_path) as conn:
#     #         cur = conn.execute(
#     #             "SELECT 1 FROM file_status_table WHERE filename = ? LIMIT 1",
#     #             (filename,),
#     #         )
#     #         return cur.fetchone() is not None


import sqlite3
from typing import Optional
from datetime import datetime
from config import STATUS_TYPE as Status


class FileRegistry:
    def __init__(self, db_path: str = "rag_registry.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Added 'domain' and 'published_at' columns
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_status_table (
                    filename TEXT PRIMARY KEY,
                    domain TEXT,
                    status TEXT CHECK(status IN ('downloaded', 'parsed', 'processed', 'embedded', 'error')),
                    error_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    published_at TIMESTAMP, 
                    downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    parsed_at TIMESTAMP,
                    processed_at TIMESTAMP,
                    embedded_at TIMESTAMP
                );
            """)
            # Create an index to make finding the "latest date" fast
            conn.execute("CREATE INDEX IF NOT EXISTS idx_domain_date ON file_status_table (domain, published_at);")

    def get_last_domain_date(self, domain: str) -> Optional[str]:
        """Returns the newest published_date we have on record for a specific domain."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT MAX(published_at) FROM file_status_table WHERE domain = ? AND status != 'error'",
                (domain,)
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] else None

    def update_status(self, filename: str, status: Status, domain: Optional[str] = None, 
                      published_at: Optional[datetime] = None, error_msg: Optional[str] = None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                if status == "error":
                    conn.execute("""
                        INSERT OR REPLACE INTO file_status_table 
                        (filename, domain, status, error_count, last_error, parsed_at, embedded_at) 
                        VALUES (?, ?, ?, 
                            COALESCE((SELECT error_count + 1 FROM file_status_table WHERE filename = ?), 1),
                            ?, NULL, NULL)
                    """, (filename, domain, status, filename, error_msg))
                
                elif status == "downloaded":
                    # We accept domain and published_at only on the initial download insert
                    conn.execute("""
                        INSERT OR IGNORE INTO file_status_table 
                        (filename, domain, status, published_at, error_count) 
                        VALUES (?, ?, ?, ?, 0)
                    """, (filename, domain, status, published_at))
                
                elif status == "parsed":
                    conn.execute("UPDATE file_status_table SET status = ?, parsed_at = CURRENT_TIMESTAMP, error_count = 0 WHERE filename = ?", 
                                (status, filename))
                    
                elif status == "processed":
                    conn.execute("UPDATE file_status_table SET status = ?, processed_at = CURRENT_TIMESTAMP, error_count = 0 WHERE filename = ?", 
                                (status, filename))
                
                elif status == "embedded":
                    conn.execute("UPDATE file_status_table SET status = ?, embedded_at = CURRENT_TIMESTAMP, error_count = 0 WHERE filename = ?", 
                                (status, filename))
                conn.commit()
            return True
        except Exception as e:
            print(f"DB Error: {e}")
            return False

    def get_status(self, filename: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT status FROM file_status_table WHERE filename = ?", 
                (filename,)
            )
            result = cursor.fetchone()
            return result[0] if result else None