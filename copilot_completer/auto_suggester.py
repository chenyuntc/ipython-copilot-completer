from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from prompt_toolkit.auto_suggest import AutoSuggest, AutoSuggestFromHistory, Suggestion
from typing_extensions import override

from .completer import fetch_copilot_suggestion


if TYPE_CHECKING:
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document


class HybridCopilotSuggest(AutoSuggest):
    """
    A hybrid auto-suggester that first tries history-based suggestions,
    and only falls back to Copilot when no history match is found.
    """

    def __init__(self):
        super().__init__()
        self.history_suggester = AutoSuggestFromHistory()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        self.loop = loop
        self.last_text = ""
        self.debounce_time = 1.0  # 1000 milliseconds
        self.pending_task = None

    @override
    def get_suggestion(
        self,
        buffer: Buffer,
        document: Document,
    ) -> Suggestion | None:
        # First, try to get a suggestion from history
        history_suggestion = self.history_suggester.get_suggestion(buffer, document)

        # If we have a history suggestion, return it immediately
        if history_suggestion:
            # Cancel any pending Copilot fetch
            if self.pending_task and not self.pending_task.done():
                self.pending_task.cancel()
            return history_suggestion

        # No history suggestion, try Copilot
        # Consider only the last line for the suggestion.
        text = document.text.rsplit("\n", 1)[-1].strip()

        # Only proceed if text is not empty and has minimum length
        if text and len(text) >= 3:
            self.last_text = text
            # Cancel previous pending task if exists
            if self.pending_task and not self.pending_task.done():
                self.pending_task.cancel()
            self.pending_task = asyncio.ensure_future(self.debounce_fetch(buffer, text))

        return None

    async def debounce_fetch(self, buffer: Buffer, text: str):
        try:
            await asyncio.sleep(self.debounce_time)
            if text == self.last_text and buffer.suggestion is None:
                # Check if text is unchanged and no history suggestion exists
                suggestion = await fetch_copilot_suggestion(buffer)
                if suggestion:
                    buffer.suggestion = Suggestion(suggestion)
                    buffer.on_suggestion_set.fire()
        except asyncio.CancelledError:
            pass  # Task was cancelled, which is expected



def enable_copilot_suggester(ipython: TerminalInteractiveShell):
    if getattr(ipython, "pt_app", None) and ipython.pt_app:
        ipython.autosuggestions_provider = None
        ipython.pt_app.auto_suggest = HybridCopilotSuggest()


def disable_copilot_suggester(ipython: TerminalInteractiveShell):
    # Revert to auto-suggesting from history
    if getattr(ipython, "pt_app", None):
        ipython.autosuggestions_provider = "NavigableAutoSuggestFromHistory"
