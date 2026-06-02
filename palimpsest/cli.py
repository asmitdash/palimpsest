"""palimpsest CLI — minimal Day-1 surface.

Examples:
  palimpsest write "User lives in Berlin" --subject user
  palimpsest read  "where does the user live?" --k 5
  palimpsest stats
  palimpsest lineage <atom-id>
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table

from palimpsest import Memory


app = typer.Typer(help="palimpsest — contradiction-aware agent memory.")
console = Console()


def _open(db: Path) -> Memory:
    return Memory.open(db)


@app.command()
def write(
    content: str,
    db: Path = typer.Option(Path("palimpsest.db"), "--db"),
    kind: str = typer.Option("semantic", "--kind"),
    subject: str = typer.Option("", "--subject", help="Optional; inferred via LLM if omitted."),
    source: str = typer.Option("", "--source"),
) -> None:
    with _open(db) as mem:
        atom_id = mem.write(content, kind=kind, subject=subject or None, source=source or None)
        console.print(f"[green]wrote[/green] {atom_id}")


@app.command()
def read(
    query: str,
    db: Path = typer.Option(Path("palimpsest.db"), "--db"),
    k: int = typer.Option(8, "--k"),
    subject: str = typer.Option("", "--subject"),
    kind: str = typer.Option("", "--kind"),
) -> None:
    with _open(db) as mem:
        hits = mem.read(query, k=k, subject=subject or None, kind=kind or None)
    if not hits:
        console.print("[yellow]no hits[/yellow]")
        return
    table = Table(title=f"top {len(hits)} for {query!r}")
    table.add_column("dist")
    table.add_column("subject")
    table.add_column("kind")
    table.add_column("content")
    for a, d in hits:
        table.add_row(f"{d:.3f}", a.subject, a.kind, a.content)
    console.print(table)


@app.command()
def lineage(atom_id: str, db: Path = typer.Option(Path("palimpsest.db"), "--db")) -> None:
    with _open(db) as mem:
        chain = mem.lineage(UUID(atom_id))
    if not chain:
        console.print("[yellow]no atom[/yellow]")
        return
    for i, a in enumerate(chain):
        marker = "→" if i == len(chain) - 1 else " "
        # Disable rich markup parsing so '[superseded]' is rendered literally
        # rather than being interpreted as a style tag.
        console.print(f"{marker} {a.id} [{a.status}] {a.content}", markup=False)


@app.command()
def stats(db: Path = typer.Option(Path("palimpsest.db"), "--db")) -> None:
    with _open(db) as mem:
        console.print(json.dumps(mem.stats(), indent=2))


if __name__ == "__main__":
    app()
