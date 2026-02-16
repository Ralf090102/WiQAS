<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RAGResponse, type RetrievalResult } from '$lib/api';
	import CarbonSend from '~icons/carbon/send-alt';
	import CarbonReset from '~icons/carbon/reset';
	import CarbonDocument from '~icons/carbon/document';

	let query = $state('');
	let loading = $state(false);
	let response: RAGResponse | null = $state(null);
	let error = $state('');

	// Settings
	let k = $state(5);
	let includeSourcesOption = $state(true);
	let enableReranking = $state(true);
	let enableQueryDecomposition = $state(false);

	async function handleSubmit() {
		if (!query.trim() || loading) return;

		loading = true;
		error = '';
		response = null;

		try {
			response = await api.rag.ask({
				query: query.trim(),
				k,
				include_sources: includeSourcesOption,
				enable_reranking: enableReranking,
				enable_query_decomposition: enableQueryDecomposition,
			});
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to get answer';
		} finally {
			loading = false;
		}
	}

	function handleReset() {
		query = '';
		response = null;
		error = '';
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			handleSubmit();
		}
	}

	// Example queries
	const examples = [
		'What is bayanihan?',
		'Ano ang pakikisama?',
		'Explain Filipino hospitality',
		'Paano magpakita ng paggalang?',
	];

	function useExample(example: string) {
		query = example;
		handleSubmit();
	}
</script>

<svelte:head>
	<title>WiQAS - Question Answering System</title>
</svelte:head>

<div class="flex h-full flex-col overflow-hidden bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50 dark:from-gray-950 dark:via-blue-950/20 dark:to-gray-950">
	<!-- Header -->
	<div class="border-b border-blue-100/50 bg-white/80 px-6 py-6 shadow-sm backdrop-blur-sm dark:border-gray-800/50 dark:bg-gray-900/80">
		<div class="mx-auto max-w-4xl">
			<div class="flex items-center gap-3">
				<div class="flex size-12 items-center justify-center rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 shadow-lg shadow-blue-500/30">
					<span class="text-2xl font-bold text-white">W</span>
				</div>
				<div>
					<h1 class="text-2xl font-bold bg-gradient-to-r from-blue-700 to-indigo-700 bg-clip-text text-transparent dark:from-blue-400 dark:to-indigo-400">WiQAS</h1>
					<p class="text-sm font-medium text-gray-600 dark:text-gray-400">
						Filipino-English Question Answering System
					</p>
				</div>
			</div>
		</div>
	</div>

	<!-- Main Content -->
	<div class="flex-1 overflow-y-auto px-6 py-8">
		<div class="mx-auto max-w-4xl space-y-6">
			<!-- Welcome / Examples (shown when no response) -->
			{#if !response && !loading}
				<div class="text-center">
					<div class="mb-12 mt-8">
						<h2 class="mb-3 text-4xl font-bold text-gray-900 dark:text-white">
							Ask me anything
						</h2>
						<p class="text-lg text-gray-600 dark:text-gray-400">
							I can answer questions in <span class="font-semibold text-blue-600 dark:text-blue-400">English</span> or <span class="font-semibold text-indigo-600 dark:text-indigo-400">Filipino</span> using the knowledge base
						</p>
					</div>

					<div class="mb-8">
						<h3 class="mb-4 text-sm font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
							Try these examples:
						</h3>
						<div class="flex flex-wrap justify-center gap-3">
							{#each examples as example}
								<button
									on:click={() => useExample(example)}
									class="group rounded-xl border border-blue-200 bg-white px-5 py-3 text-sm font-medium text-gray-700 shadow-sm transition-all hover:scale-105 hover:border-blue-300 hover:bg-blue-50 hover:shadow-md dark:border-gray-700 dark:bg-gray-800/50 dark:text-gray-300 dark:hover:border-blue-700 dark:hover:bg-gray-700"
								>
									{example}
								</button>
							{/each}
						</div>
					</div>
				</div>
			{/if}

			<!-- Error Message -->
			{#if error}
				<div class="rounded-xl border border-red-200 bg-red-50 p-5 shadow-sm dark:border-red-800 dark:bg-red-900/20">
					<p class="text-sm text-red-800 dark:text-red-300">
						<strong class="font-semibold">Error:</strong> {error}
					</p>
				</div>
			{/if}

			<!-- Response -->
			{#if response}
				<div class="space-y-6">
					<!-- Query Info -->
				<div class="rounded-xl border border-blue-200 bg-gradient-to-br from-blue-50 to-indigo-50 p-5 shadow-sm dark:border-blue-800 dark:from-blue-950/40 dark:to-indigo-950/40">
					<div class="flex items-start gap-4">
						<div class="rounded-xl bg-gradient-to-br from-blue-500 to-indigo-500 p-2.5 shadow-md">
							<svg class="size-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
							</svg>
						</div>
						<div class="flex-1">
							<p class="font-semibold text-blue-900 dark:text-blue-100">
								{response.query}
							</p>
							{#if response.detected_language}
								<p class="mt-2 inline-block rounded-full bg-100 px-3 py-1 text-xs font-medium text-blue-700 dark:bg-blue-900/50 dark:text-blue-300">
									{response.detected_language === 'fil' ? '🇵🇭 Filipino' : '🇬🇧 English'}
								</p>
							{/if}
							</div>
						</div>
					</div>

					<!-- Answer -->
				<div class="rounded-xl border border-gray-200 bg-white p-6 shadow-md dark:border-gray-700 dark:bg-gray-800">
					<h3 class="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
						<div class="size-1.5 rounded-full bg-green-500"></div>
						Answer
					</h3>
					<div class="prose prose-sm dark:prose-invert max-w-none">
						<p class="whitespace-pre-wrap text-base leading-relaxed text-gray-800 dark:text-gray-200">
							{response.answer}
						</p>
						</div>
					</div>

					<!-- Sources -->
					{#if response.sources && response.sources.length > 0}
					<div class="rounded-xl border border-gray-200 bg-white p-6 shadow-md dark:border-gray-700 dark:bg-gray-800">
						<h3 class="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
							<CarbonDocument class="size-4" />
							Sources ({response.total_sources})
						</h3>
						<div class="space-y-3">
							{#each response.sources as source, idx}
								<div class="group rounded-xl border border-gray-200 bg-gradient-to-br from-gray-50 to-slate-50 p-4 transition-all hover:border-blue-200 hover:shadow-md dark:border-gray-700 dark:from-gray-900 dark:to-gray-800 dark:hover:border-blue-800">
									<div class="mb-3 flex items-start justify-between">
										<div class="flex items-center gap-2">
											<div class="rounded-lg bg-blue-100 p-1.5 dark:bg-blue-900/50">
												<CarbonDocument class="size-3.5 text-blue-600 dark:text-blue-400" />
											</div>
											<span class="text-xs font-semibold text-gray-700 dark:text-gray-300">
												Source {idx + 1}
											</span>
										</div>
										<span class="rounded-full bg-gray-200 px-2 py-0.5 text-xs font-medium text-gray-600 dark:bg-gray-700 dark:text-gray-400">
											{source.score.toFixed(4)}
										</span>
									</div>
									<p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
										{source.content}
									</p>
									{#if source.metadata.source}
										<p class="mt-3 flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
											<span class="rounded bg-gray-200 px-1.5 py-0.5 font-medium dark:bg-gray-700">{source.metadata.source}</span>
											{#if source.metadata.page}
												<span>·</span>
												<span>Page {source.metadata.page}</span>
											{/if}
										</p>
									{/if}
									</div>
								{/each}
							</div>
						</div>
					{/if}

					<!-- Timing Info -->
					{#if response.timing}
					<div class="rounded-xl border border-gray-200 bg-gradient-to-br from-slate-50 to-gray-50 p-5 shadow-sm dark:border-gray-700 dark:from-gray-900 dark:to-gray-800">
						<h3 class="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
							<svg class="size-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
							</svg>
								Performance
							</h3>
							<div class="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
								{#if response.timing.query_decomposition_time}
									<div>
										<div class="text-gray-500">Decomposition</div>
										<div class="font-mono font-semibold text-gray-900 dark:text-white">
											{response.timing.query_decomposition_time.toFixed(3)}s
										</div>
									</div>
								{/if}
								<div>
									<div class="text-gray-500">Retrieval</div>
									<div class="font-mono font-semibold text-gray-900 dark:text-white">
										{response.timing.retrieval_time.toFixed(3)}s
									</div>
								</div>
								{#if response.timing.reranking_time}
									<div>
										<div class="text-gray-500">Reranking</div>
										<div class="font-mono font-semibold text-gray-900 dark:text-white">
											{response.timing.reranking_time.toFixed(3)}s
										</div>
									</div>
								{/if}
								<div>
									<div class="text-gray-500">Generation</div>
									<div class="font-mono font-semibold text-gray-900 dark:text-white">
										{response.timing.generation_time.toFixed(3)}s
									</div>
								</div>
								<div class="font-semibold">
									<div class="text-gray-500">Total</div>
									<div class="font-mono text-blue-600 dark:text-blue-400">
										{response.timing.total_time.toFixed(3)}s
									</div>
								</div>
							</div>
						</div>
					{/if}

					<!-- Reset Button -->
					<div class="flex justify-center">
						<button
							on:click={handleReset}
							class="flex items-center gap-2 rounded-xl border border-blue-200 bg-gradient-to-r from-white to-blue-50 px-6 py-3 text-sm font-medium text-gray-700 shadow-sm transition-all hover:scale-105 hover:border-blue-300 hover:shadow-md dark:border-gray-700 dark:from-gray-800 dark:to-gray-700 dark:text-gray-300 dark:hover:border-blue-700"
						>
							<CarbonReset class="size-4" />
							Ask Another Question
						</button>
					</div>
				</div>
			{/if}

			<!-- Loading State -->
			{#if loading}
				<div class="flex flex-col items-center justify-center py-16">
					<div class="relative mb-6">
						<div class="size-16 animate-spin rounded-full border-4 border-blue-200 border-t-blue-600 dark:border-blue-900 dark:border-t-blue-400"></div>
						<div class="absolute inset-0 flex items-center justify-center">
							<div class="size-8 animate-pulse rounded-full bg-blue-100 dark:bg-blue-900"></div>
						</div>
					</div>
					<p class="text-base font-medium text-gray-700 dark:text-gray-300">
						Searching knowledge base...
					</p>
					<p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
						Generating your answer
					</p>
				</div>
			{/if}
		</div>
	</div>

	<!-- Input Area (Fixed at bottom) -->
	<div class="border-t border-blue-100/50 bg-white/90 px-6 py-5 shadow-lg backdrop-blur-sm dark:border-gray-800/50 dark:bg-gray-900/90">
		<div class="mx-auto max-w-4xl">
			<div class="flex gap-3">
				<div class="relative flex-1">
					<textarea
						bind:value={query}
						on:keydown={handleKeydown}
						placeholder="Ask a question in English or Filipino..."
						rows="2"
						disabled={loading}
						class="w-full resize-none rounded-xl border-2 border-gray-200 bg-white px-5 py-4 text-gray-900 placeholder-gray-400 shadow-sm transition-all focus:border-blue-500 focus:outline-none focus:ring-4 focus:ring-blue-500/10 disabled:opacity-50 dark:border-gray-700 dark:bg-gray-800 dark:text-white dark:placeholder-gray-500 dark:focus:border-blue-600"
					></textarea>
				</div>
				<button
					on:click={handleSubmit}
					disabled={loading || !query.trim()}
					class="flex h-full min-w-[4rem] items-center justify-center rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-6 text-white shadow-lg shadow-blue-500/30 transition-all hover:scale-105 hover:from-blue-700 hover:to-indigo-700 hover:shadow-xl hover:shadow-blue-500/40 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100"
				>
					{#if loading}
						<div class="size-5 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
					{:else}
						<CarbonSend class="size-5" />
					{/if}
				</button>
			</div>
			<p class="mt-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400">
				Press <kbd class="rounded bg-gray-200 px-1.5 py-0.5 font-mono text-gray-700 dark:bg-gray-700 dark:text-gray-300">Enter</kbd> to send · <kbd class="rounded bg-gray-200 px-1.5 py-0.5 font-mono text-gray-700 dark:bg-gray-700 dark:text-gray-300">Shift+Enter</kbd> for new line
			</p>
		</div>
	</div>
</div>

