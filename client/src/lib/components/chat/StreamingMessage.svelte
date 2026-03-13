<script lang="ts">
  import { chatStore } from "$lib/stores";
  import { fade } from "svelte/transition";
  import MarkdownRenderer from "./MarkdownRenderer.svelte";

  let streamingContent = $derived(chatStore.streamingContent);
  let hasContent = $derived(streamingContent.length > 0);
  let statusText = $derived(chatStore.streamingStatus);
</script>

<div class="flex flex-col gap-1">
  <div class="flex items-center gap-2">
    <span class="text-sm">🐾</span>
    <span class="text-xs font-medium text-muted-foreground">PocketPaw</span>
  </div>

  <div class="max-w-full pl-6">
    {#if hasContent}
      <div class="text-sm leading-relaxed text-foreground">
        <MarkdownRenderer content={streamingContent} />
      </div>
    {/if}

    {#if statusText}
      <div
        class="group relative flex items-center gap-3 overflow-hidden rounded-full border border-border/50 bg-card/40 backdrop-blur-md px-3 py-1.5 {hasContent
          ? 'mt-4'
          : 'mt-1'} max-w-fit shadow-sm transition-all duration-500 hover:border-paw-accent/40 hover:bg-card/60 hover:shadow-paw-accent/5"
        transition:fade={{ duration: 200 }}
      >
        <!-- Pulsing Status Indicator -->
        <div class="relative flex h-2 w-2">
          <div
            class="absolute inline-flex h-full w-full animate-ping rounded-full bg-paw-accent/50 opacity-75"
          ></div>
          <div
            class="relative inline-flex h-2 w-2 rounded-full bg-paw-accent shadow-[0_0_8px_rgba(255,160,0,0.4)]"
          ></div>
        </div>

        <!-- Status text with cursor -->
        <div class="flex items-center gap-1.5 min-w-0">
          <span class="text-[11px] font-semibold tracking-wide text-foreground/70 uppercase">
            {statusText}
          </span>
          <span class="h-3 w-[1.5px] animate-cursor-blink rounded-full bg-paw-accent/60"></span>
        </div>

        <!-- Shimmer Effect -->
        <div
          class="pointer-events-none absolute inset-y-0 -left-[100%] w-1/2 animate-progress-slide skew-x-12 bg-gradient-to-r from-transparent via-white/10 to-transparent opacity-30"
        ></div>
      </div>
    {/if}
  </div>
</div>
