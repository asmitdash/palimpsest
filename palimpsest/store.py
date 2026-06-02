"""SQLite + sqlite-vec storage.

Schema:
  atoms                  — one row per memory atom (immutable except status)
  atom_embeddings        — sqlite-vec virtual table, vector index
  reinforcements         — append-only log of every re-assertion of an atom
  contradictions         — append-only log of detected contradictions + resolutions

Why two-table embeddings: sqlite-vec virtual tables don't store metadata, only
vectors keyed by rowid. We map atom UUID -> rowid via a thin index column.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from palimpsest.schemas import Atom, AtomKind, AtomStatus


_VEC_DIM_DEFAULT = 768  # google-genai text-embedding-004; configurable per-instance


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _floats_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_floats(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class Store:
    """SQLite-backed atom store. Connection is held for the lifetime of the
    Store; use one Store per memory namespace (db file).

    Embedding dimension is fixed at open time (sqlite-vec requirement) and
    persisted in a meta table so re-opens validate consistency.
    """

    def __init__(self, db_path: str | Path, *, vec_dim: int = _VEC_DIM_DEFAULT) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._load_vec_extension()
        self.vec_dim = self._init_schema(vec_dim)

    # ---------- setup ----------

    def _load_vec_extension(self) -> None:
        try:
            import sqlite_vec  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "sqlite-vec is required. `pip install sqlite-vec`."
            ) from e
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

    def _init_schema(self, requested_dim: int) -> int:
        c = self.conn.cursor()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS atoms (
                id                  TEXT PRIMARY KEY,
                content             TEXT NOT NULL,
                kind                TEXT NOT NULL CHECK (kind IN ('episodic', 'semantic', 'procedural')),
                subject             TEXT NOT NULL,
                source              TEXT,
                confidence          REAL NOT NULL DEFAULT 1.0,
                status              TEXT NOT NULL DEFAULT 'active'
                                      CHECK (status IN ('active', 'superseded', 'contradicted', 'retracted')),
                supersedes_id       TEXT REFERENCES atoms(id),
                superseded_by_id    TEXT REFERENCES atoms(id),
                created_at          TEXT NOT NULL,
                last_reinforced_at  TEXT NOT NULL,
                reinforcement_count INTEGER NOT NULL DEFAULT 1,
                metadata            TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_atoms_subject ON atoms(subject);
            CREATE INDEX IF NOT EXISTS idx_atoms_kind ON atoms(kind);
            CREATE INDEX IF NOT EXISTS idx_atoms_status ON atoms(status);

            CREATE TABLE IF NOT EXISTS reinforcements (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                atom_id     TEXT NOT NULL REFERENCES atoms(id),
                source      TEXT,
                confidence  REAL NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reinforcements_atom ON reinforcements(atom_id);

            CREATE TABLE IF NOT EXISTS contradictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                new_atom_id     TEXT NOT NULL REFERENCES atoms(id),
                prior_atom_id   TEXT NOT NULL REFERENCES atoms(id),
                severity        TEXT NOT NULL,
                verifier_rationale TEXT NOT NULL,
                resolution_action  TEXT NOT NULL,
                resolution_rationale TEXT NOT NULL,
                merged_atom_id  TEXT REFERENCES atoms(id),
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_contradictions_new ON contradictions(new_atom_id);
            CREATE INDEX IF NOT EXISTS idx_contradictions_prior ON contradictions(prior_atom_id);

            -- atom_id -> integer rowid mapping for the vec virtual table
            CREATE TABLE IF NOT EXISTS atom_vec_map (
                atom_id  TEXT PRIMARY KEY REFERENCES atoms(id),
                rowid    INTEGER NOT NULL UNIQUE
            );
            """
        )

        # vec virtual table dimension is fixed at create time; reconcile with meta.
        existing = c.execute("SELECT value FROM meta WHERE key='vec_dim'").fetchone()
        if existing is None:
            dim = requested_dim
            c.execute(
                "INSERT INTO meta(key, value) VALUES ('vec_dim', ?)", (str(dim),)
            )
        else:
            dim = int(existing["value"])
            if dim != requested_dim:
                # Caller asked for a different dim than the file already uses.
                # We honour the file. Caller is responsible for consistent embedders.
                pass

        c.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS atom_embeddings USING vec0(embedding float[{dim}])"
        )
        self.conn.commit()
        return dim

    # ---------- writes ----------

    def insert_atom(self, atom: Atom, embedding: list[float]) -> None:
        if len(embedding) != self.vec_dim:
            raise ValueError(
                f"embedding dim {len(embedding)} != store dim {self.vec_dim}"
            )
        c = self.conn.cursor()
        c.execute(
            """
            INSERT INTO atoms
              (id, content, kind, subject, source, confidence, status,
               supersedes_id, superseded_by_id,
               created_at, last_reinforced_at, reinforcement_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')
            """,
            (
                str(atom.id), atom.content, atom.kind, atom.subject, atom.source,
                atom.confidence, atom.status,
                str(atom.supersedes_id) if atom.supersedes_id else None,
                str(atom.superseded_by_id) if atom.superseded_by_id else None,
                atom.created_at.isoformat(),
                atom.last_reinforced_at.isoformat(),
                atom.reinforcement_count,
            ),
        )
        # vec virtual table requires sequential rowids; we let it auto-assign
        # then capture last_insert_rowid for our map.
        c.execute(
            "INSERT INTO atom_embeddings(embedding) VALUES (?)",
            (_floats_to_blob(embedding),),
        )
        rowid = c.lastrowid
        c.execute(
            "INSERT INTO atom_vec_map(atom_id, rowid) VALUES (?, ?)",
            (str(atom.id), rowid),
        )
        self.conn.commit()

    def mark_superseded(self, old_id: UUID, new_id: UUID) -> None:
        c = self.conn.cursor()
        c.execute(
            """
            UPDATE atoms
               SET status='superseded', superseded_by_id=?
             WHERE id=? AND status='active'
            """,
            (str(new_id), str(old_id)),
        )
        c.execute(
            "UPDATE atoms SET supersedes_id=? WHERE id=?",
            (str(old_id), str(new_id)),
        )
        self.conn.commit()

    def mark_contradicted(self, atom_id: UUID) -> None:
        self.conn.execute(
            "UPDATE atoms SET status='contradicted' WHERE id=?",
            (str(atom_id),),
        )
        self.conn.commit()

    def retract(self, atom_id: UUID) -> None:
        self.conn.execute(
            "UPDATE atoms SET status='retracted' WHERE id=?",
            (str(atom_id),),
        )
        self.conn.commit()

    def record_reinforcement(
        self,
        atom_id: UUID,
        *,
        source: str | None,
        confidence: float,
        new_confidence: float | None = None,
    ) -> None:
        """Append a reinforcement row + bump counters.

        If `new_confidence` is provided, it is set verbatim (use this when the
        confidence engine has already computed the right value). Otherwise we
        do the legacy +0.05*confidence fallback.
        """
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO reinforcements(atom_id, source, confidence, created_at) VALUES (?, ?, ?, ?)",
            (str(atom_id), source, confidence, _utcnow_iso()),
        )
        if new_confidence is None:
            c.execute(
                """
                UPDATE atoms
                   SET reinforcement_count = reinforcement_count + 1,
                       last_reinforced_at  = ?,
                       confidence          = MIN(1.0, confidence + ?)
                 WHERE id = ?
                """,
                (_utcnow_iso(), 0.05 * confidence, str(atom_id)),
            )
        else:
            c.execute(
                """
                UPDATE atoms
                   SET reinforcement_count = reinforcement_count + 1,
                       last_reinforced_at  = ?,
                       confidence          = ?
                 WHERE id = ?
                """,
                (_utcnow_iso(), max(0.0, min(1.0, new_confidence)), str(atom_id)),
            )
        self.conn.commit()

    def set_confidence(self, atom_id: UUID, confidence: float) -> None:
        self.conn.execute(
            "UPDATE atoms SET confidence = ? WHERE id = ?",
            (max(0.0, min(1.0, confidence)), str(atom_id)),
        )
        self.conn.commit()

    def record_contradiction(
        self,
        *,
        new_atom_id: UUID,
        prior_atom_id: UUID,
        severity: str,
        verifier_rationale: str,
        resolution_action: str,
        resolution_rationale: str,
        merged_atom_id: UUID | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO contradictions
              (new_atom_id, prior_atom_id, severity, verifier_rationale,
               resolution_action, resolution_rationale, merged_atom_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(new_atom_id), str(prior_atom_id), severity, verifier_rationale,
                resolution_action, resolution_rationale,
                str(merged_atom_id) if merged_atom_id else None,
                _utcnow_iso(),
            ),
        )
        self.conn.commit()

    # ---------- reads ----------

    def get_atom(self, atom_id: UUID) -> Atom | None:
        row = self.conn.execute(
            "SELECT * FROM atoms WHERE id=?", (str(atom_id),)
        ).fetchone()
        return _row_to_atom(row) if row else None

    def search(
        self,
        query_embedding: list[float],
        *,
        k: int = 10,
        subject: str | None = None,
        kind: AtomKind | None = None,
        status: AtomStatus | tuple[AtomStatus, ...] | None = "active",
        as_of: datetime | None = None,
    ) -> list[tuple[Atom, float]]:
        """Vector retrieval with status / subject / kind / as-of filters.

        Returns (atom, distance) tuples. Lower distance = closer match.
        """
        if len(query_embedding) != self.vec_dim:
            raise ValueError(
                f"query dim {len(query_embedding)} != store dim {self.vec_dim}"
            )

        # vec_search returns rowids ranked by distance; we then join out via the map.
        # Pull top (k * 4) candidates from the index then filter by metadata so the
        # final list reaches `k` after filtering.
        candidate_n = max(k * 4, 32)
        rows = self.conn.execute(
            """
            SELECT av.atom_id AS atom_id, ae.distance AS distance
            FROM atom_embeddings ae
            JOIN atom_vec_map av ON av.rowid = ae.rowid
            WHERE ae.embedding MATCH ?
              AND k = ?
            ORDER BY ae.distance
            """,
            (_floats_to_blob(query_embedding), candidate_n),
        ).fetchall()

        if not rows:
            return []

        ids = [r["atom_id"] for r in rows]
        dist_by_id = {r["atom_id"]: r["distance"] for r in rows}

        placeholders = ",".join("?" for _ in ids)
        sql = f"SELECT * FROM atoms WHERE id IN ({placeholders})"
        params: list[Any] = list(ids)

        if subject is not None:
            sql += " AND subject = ?"
            params.append(subject.lower().strip())
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        if status is not None:
            if isinstance(status, str):
                statuses = (status,)
            else:
                statuses = tuple(status)
            sql += f" AND status IN ({','.join('?' for _ in statuses)})"
            params.extend(statuses)
        if as_of is not None:
            sql += " AND created_at <= ?"
            params.append(as_of.isoformat())

        atoms_rows = self.conn.execute(sql, params).fetchall()

        ranked = sorted(
            (_row_to_atom(r) for r in atoms_rows),
            key=lambda a: dist_by_id.get(str(a.id), float("inf")),
        )
        return [(a, dist_by_id[str(a.id)]) for a in ranked[:k]]

    def list_by_subject(self, subject: str, *, status: AtomStatus | None = "active") -> list[Atom]:
        sql = "SELECT * FROM atoms WHERE subject = ?"
        params: list[Any] = [subject.lower().strip()]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY last_reinforced_at DESC"
        rows = self.conn.execute(sql, params).fetchall()
        return [_row_to_atom(r) for r in rows]

    def all_active_subjects(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT subject FROM atoms WHERE status='active'"
        ).fetchall()
        return [r["subject"] for r in rows]

    def list_active(
        self, *, kind: AtomKind | None = None, limit: int = 1000,
    ) -> list[Atom]:
        sql = "SELECT * FROM atoms WHERE status='active'"
        params: list[Any] = []
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY last_reinforced_at ASC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [_row_to_atom(r) for r in rows]

    def episodic_window(
        self,
        subject: str | None,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Atom]:
        """Episodic-specific retrieval: ordered by created_at, optionally
        bounded by a [start, end] window."""
        sql = "SELECT * FROM atoms WHERE kind='episodic' AND status='active'"
        params: list[Any] = []
        if subject is not None:
            sql += " AND subject = ?"
            params.append(subject.lower().strip())
        if start is not None:
            sql += " AND created_at >= ?"
            params.append(start.isoformat())
        if end is not None:
            sql += " AND created_at <= ?"
            params.append(end.isoformat())
        sql += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [_row_to_atom(r) for r in rows]

    def lineage_chain(self, atom_id: UUID) -> list[Atom]:
        """Walk supersedes_id backwards: returns [oldest, ..., this_atom]."""
        chain: list[Atom] = []
        cur_id: str | None = str(atom_id)
        seen: set[str] = set()
        while cur_id and cur_id not in seen:
            seen.add(cur_id)
            row = self.conn.execute("SELECT * FROM atoms WHERE id=?", (cur_id,)).fetchone()
            if not row:
                break
            chain.append(_row_to_atom(row))
            cur_id = row["supersedes_id"]
        return list(reversed(chain))

    def stats(self) -> dict[str, Any]:
        c = self.conn.cursor()
        return {
            "atoms_total": c.execute("SELECT COUNT(*) FROM atoms").fetchone()[0],
            "atoms_active": c.execute("SELECT COUNT(*) FROM atoms WHERE status='active'").fetchone()[0],
            "atoms_superseded": c.execute("SELECT COUNT(*) FROM atoms WHERE status='superseded'").fetchone()[0],
            "atoms_contradicted": c.execute("SELECT COUNT(*) FROM atoms WHERE status='contradicted'").fetchone()[0],
            "contradictions_logged": c.execute("SELECT COUNT(*) FROM contradictions").fetchone()[0],
            "reinforcements_logged": c.execute("SELECT COUNT(*) FROM reinforcements").fetchone()[0],
            "vec_dim": self.vec_dim,
        }

    def close(self) -> None:
        self.conn.close()


def _row_to_atom(row: sqlite3.Row) -> Atom:
    return Atom(
        id=UUID(row["id"]),
        content=row["content"],
        kind=row["kind"],
        subject=row["subject"],
        source=row["source"],
        confidence=row["confidence"],
        status=row["status"],
        supersedes_id=UUID(row["supersedes_id"]) if row["supersedes_id"] else None,
        superseded_by_id=UUID(row["superseded_by_id"]) if row["superseded_by_id"] else None,
        created_at=datetime.fromisoformat(row["created_at"]),
        last_reinforced_at=datetime.fromisoformat(row["last_reinforced_at"]),
        reinforcement_count=row["reinforcement_count"],
    )
