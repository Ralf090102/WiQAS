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

<div class="flex h-full flex-col overflow-hidden">
	<!-- Header -->
	<div class="border-b border-gray-200 bg-white px-6 py-4 dark:border-gray-700 dark:bg-gray-800">
		<div class="mx-auto max-w-4xl">
			<h1 class="text-2xl font-bold text-gray-900 dark:text-white">WiQAS</h1>
			<p class="text-sm text-gray-600 dark:text-gray-400">
				Filipino-English Question Answering System
			</p>
		</div>
	</div>

	<!-- Main Content -->
	<div class="flex-1 overflow-y-auto px-6 py-8">
		<div class="mx-auto max-w-4xl space-y-6">
			<!-- Welcome / Examples (shown when no response) -->
			{#if !response && !loading}
				<div class="text-center">
					<div class="mb-8">
						<h2 class="mb-2 text-3xl font-bold text-gray-900 dark:text-white">
							Ask me anything
						</h2>
						<p class="text-gray-600 dark:text-gray-400">
							I can answer questions in English or Filipino using the knowledge base
						</p>
					</div>

					<div class="mb-8">
						<h3 class="mb-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
							Try these examples:
						</h3>
						<div class="flex flex-wrap justify-center gap-2">
							{#each examples as example}
								<button
									on:click={() => useExample(example)}
									class="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:hover:bg-gray-700"
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
				<div class="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-900/20">
					<p class="text-sm text-red-800 dark:text-red-300">
						<strong>Error:</strong> {error}
					</p>
				</div>
			{/if}

			<!-- Response -->
			{#if response}
				<div class="space-y-6">
					<!-- Query Info -->
					<div class="rounded-lg border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/20">
						<div class="flex items-start gap-3">
							<div class="rounded-lg bg-blue-100 p-2 dark:bg-blue-900/40">
								<svg class="size-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
								</svg>
							</div>
							<div class="flex-1">
								<p class="font-medium text-blue-900 dark:text-blue-100">
									{response.query}
								</p>
								{#if response.detected_language}
									<p class="mt-1 text-xs text-blue-700 dark:text-blue-300">
										Language: {response.detected_language === 'fil' ? 'Filipino' : 'English'}
									</p>
								{/if}
							</div>
						</div>
					</div>

					<!-- Answer -->
					<div class="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
						<h3 class="mb-3 font-semibold text-gray-900 dark:text-white">Answer</h3>
						<div class="prose prose-sm dark:prose-invert max-w-none">
							<p class="whitespace-pre-wrap text-gray-800 dark:text-gray-200">
								{response.answer}
							</p>
						</div>
					</div>

					<!-- Sources -->
					{#if response.sources && response.sources.length > 0}
						<div class="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
							<h3 class="mb-4 font-semibold text-gray-900 dark:text-white">
								Sources ({response.total_sources})
							</h3>
							<div class="space-y-3">
								{#each response.sources as source, idx}
									<div class="rounded-lg border border-gray-200 bg-gray-50 p-4 dark:border-gray-600 dark:bg-gray-700/50">
										<div class="mb-2 flex items-start justify-between">
											<div class="flex items-center gap-2">
												<CarbonDocument class="size-4 text-gray-500" />
												<span class="text-xs font-semibold text-gray-600 dark:text-gray-400">
													Source {idx + 1}
												</span>
											</div>
											<span class="text-xs text-gray-500">
												Score: {source.score.toFixed(4)}
											</span>
										</div>
										<p class="text-sm text-gray-700 dark:text-gray-300">
											{source.content}
										</p>
										{#if source.metadata.source}
											<p class="mt-2 text-xs text-gray-500 dark:text-gray-400">
												{source.metadata.source}
												{#if source.metadata.page}
													· Page {source.metadata.page}
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
						<div class="rounded-lg border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-800/50">
							<h3 class="mb-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
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
							class="flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:hover:bg-gray-700"
						>
							<CarbonReset class="size-4" />
							Ask Another Question
						</button>
					</div>
				</div>
			{/if}

			<!-- Loading State -->
			{#if loading}
				<div class="flex flex-col items-center justify-center py-12">
					<div class="mb-4 size-12 animate-spin rounded-full border-4 border-gray-200 border-t-blue-600"></div>
					<p class="text-sm text-gray-600 dark:text-gray-400">
						Searching knowledge base and generating answer...
					</p>
				</div>
			{/if}
		</div>
	</div>

	<!-- Input Area (Fixed at bottom) -->
	<div class="border-t border-gray-200 bg-white px-6 py-4 dark:border-gray-700 dark:bg-gray-800">
		<div class="mx-auto max-w-4xl">
			<div class="flex gap-3">
				<textarea
					bind:value={query}
					on:keydown={handleKeydown}
					placeholder="Ask a question in English or Filipino..."
					rows="2"
					disabled={loading}
					class="flex-1 resize-none rounded-lg border border-gray-300 bg-white px-4 py-3 text-gray-900 placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:opacity-50 dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400"
				></textarea>
				<button
					on:click={handleSubmit}
					disabled={loading || !query.trim()}
					class="flex h-full items-center justify-center rounded-lg bg-blue-600 px-6 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
				>
					{#if loading}
						<div class="size-5 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
					{:else}
						<CarbonSend class="size-5" />
					{/if}
				</button>
			</div>
			<p class="mt-2 text-xs text-gray-500 dark:text-gray-400">
				Press Enter to send, Shift+Enter for new line
			</p>
		</div>
	</div>
</div>

