/**
 * Typed API client utilities for WiQAS backend
 * Simple fetch wrappers with proper error handling
 */

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

/**
 * API error class for better error handling
 */
export class APIError extends Error {
	constructor(
		message: string,
		public status: number,
		public data?: unknown
	) {
		super(message);
		this.name = 'APIError';
	}
}

/**
 * Type definitions for API responses
 */
export interface Session {
	session_id: string;
	created_at: string;
	updated_at: string;
	message_count: number;
	metadata: Record<string, unknown>;
	messages?: Array<{
		role: 'user' | 'assistant';
		content: string;
		timestamp: string;
	}>;
}

export interface SessionResponse {
	status: string;
	message: string;
	session: Session;
	session_id?: string;
}

export interface SessionListResponse {
	status: string;
	sessions: Session[];
}

/**
 * Models API types
 */
export interface ModelConfig {
	model: string;
	base_url: string;
	temperature: number;
	top_p: number;
	max_tokens: number | null;
	timeout: number;
	system_prompt: string;
}

export interface UpdateModelRequest {
	model: string;
}

export interface UpdateLLMConfigRequest {
	temperature?: number;
	top_p?: number;
	max_tokens?: number | null;
	timeout?: number;
	system_prompt?: string;
}

export interface OllamaModelInfo {
	name: string;
	id: string;
	size: string;
	size_bytes: number;
	modified: string;
	details?: {
		format?: string;
		family?: string;
		parameter_size?: string;
	};
}

export interface ModelsListResponse {
	status: string;
	current_model: string;
	total_models: number;
	models: OllamaModelInfo[];
}

/**
 * Settings API types
 */
export interface EmbeddingSettings {
	model: string;
	device: string;
	batch_size: number;
	normalize_embeddings: boolean;
	show_progress_bar: boolean;
}

export interface ChunkingSettings {
	chunk_size: number;
	chunk_overlap: number;
	length_function: string;
	separators: string[];
}

export interface RetrievalSettings {
	default_k: number;
	enable_reranking: boolean;
	enable_cross_lingual_retrieval: boolean;
	enable_query_decomposition: boolean;
	decomposition_max_subqueries: number;
	rrf_k: number;
}

export interface RerankerSettings {
	model: string;
	device: string;
	batch_size: number;
	top_k: number;
}

export interface GenerationSettings {
	mode: string;
	enable_citations: boolean;
	enable_context_validation: boolean;
	max_context_length: number;
	citation_format: string;
}

export interface VectorStoreSettings {
	collection_name: string;
	persist_directory: string;
	distance_metric: string;
}

export interface GPUSettings {
	enabled: boolean;
	auto_detect: boolean;
	preferred_device: string;
	fallback_to_cpu: boolean;
}

export interface SettingsResponse {
	embedding: EmbeddingSettings;
	chunking: ChunkingSettings;
	retrieval: RetrievalSettings;
	reranker: RerankerSettings;
	generation: GenerationSettings;
	vectorstore: VectorStoreSettings;
	gpu: GPUSettings;
	last_updated?: string;
}

export interface SettingsUpdateRequest {
	embedding?: Partial<EmbeddingSettings>;
	chunking?: Partial<ChunkingSettings>;
	retrieval?: Partial<RetrievalSettings>;
	reranker?: Partial<RerankerSettings>;
	generation?: Partial<GenerationSettings>;
	vectorstore?: Partial<VectorStoreSettings>;
	gpu?: Partial<GPUSettings>;
}

export interface SettingsUpdateResponse {
	status: string;
	message: string;
	updated_categories: string[];
	requires_restart: string[];
	warnings: string[];
	settings: SettingsResponse;
}

/**
 * RAG API types
 */
export interface DocumentMetadata {
	source: string;
	page?: number;
	file_type?: string;
	chunk_index?: number;
	[key: string]: unknown;
}

export interface RetrievalResult {
	content: string;
	metadata: DocumentMetadata;
	score: number;
	rank?: number;
}

export interface TimingBreakdown {
	language_detection_time?: number;
	translation_time?: number;
	query_decomposition_time?: number;
	retrieval_time: number;
	reranking_time?: number;
	generation_time: number;
	total_time: number;
}

export interface QueryRequest {
	query: string;
	k?: number;
	enable_reranking?: boolean;
	enable_cross_lingual_retrieval?: boolean;
	enable_query_decomposition?: boolean;
}

export interface QueryResponse {
	status: string;
	query: string;
	detected_language?: string;
	translated_query?: string;
	results: RetrievalResult[];
	total_results: number;
	timing: TimingBreakdown;
}

export interface RAGRequest {
	query: string;
	k?: number;
	include_sources?: boolean;
	enable_reranking?: boolean;
	enable_cross_lingual_retrieval?: boolean;
	enable_query_decomposition?: boolean;
}

export interface RAGResponse {
	status: string;
	query: string;
	detected_language?: string;
	translated_query?: string;
	answer: string;
	sources?: RetrievalResult[];
	total_sources?: number;
	timing: TimingBreakdown;
}

/**
 * Ingestion API types
 */
export interface IngestionStats {
	total_files: number;
	successful: number;
	failed: number;
	skipped: number;
	total_chunks: number;
	processing_time: number;
	files_by_type: Record<string, number>;
	failed_files?: string[];
}

export interface IngestRequest {
	path: string;
	clear_existing?: boolean;
	recursive?: boolean;
}

export interface IngestResponse {
	status: string;
	message: string;
	stats: IngestionStats;
	started_at: string;
	completed_at: string;
}

export interface IngestionTask {
	task_id: string;
	status: string;
	path: string;
	progress: number;
	started_at: string;
	completed_at?: string;
	stats?: IngestionStats;
	error?: string;
}

export interface IngestTaskResponse {
	task_id: string;
	status: string;
	message: string;
	check_status_url: string;
}

export interface ClearRequest {
	confirm: boolean;
}

export interface ClearResponse {
	status: string;
	message: string;
	timestamp: string;
}

export interface KnowledgeBaseStats {
	total_chunks: number;
	unique_files: number;
	collection_name: string;
	persist_directory: string;
	file_type_distribution: Record<string, number>;
	last_updated?: string;
}

export interface StatusResponse {
	status: string;
	version: string;
	knowledge_base: KnowledgeBaseStats;
	gpu_available: boolean;
	gpu_name?: string;
	ollama_available: boolean;
	embedding_model: string;
	llm_model: string;
}

export interface FormatsResponse {
	total_formats: number;
	formats_by_category: Record<string, string[]>;
	all_formats: string[];
}

/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI<T>(
	endpoint: string,
	options: RequestInit = {}
): Promise<T> {
	const url = `${BACKEND_URL}${endpoint}`;
	
	try {
		const response = await fetch(url, {
			...options,
			headers: {
				'Content-Type': 'application/json',
				...options.headers,
			},
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => null);
			throw new APIError(
				errorData?.detail || `HTTP ${response.status}: ${response.statusText}`,
				response.status,
				errorData
			);
		}

		return await response.json();
	} catch (err) {
		if (err instanceof APIError) {
			throw err;
		}
		throw new APIError(
			err instanceof Error ? err.message : 'Network error',
			0
		);
	}
}

/**
 * API client methods
 */
export const api = {
	// Models management
	models: {
		// List all available Ollama models
		list: () =>
			fetchAPI<ModelsListResponse>('/api/models'),

		// Get current LLM configuration
		getConfig: () =>
			fetchAPI<ModelConfig>('/api/models/config'),

		// Update active model
		updateModel: (model: string) =>
			fetchAPI<ModelConfig>('/api/models/config', {
				method: 'PATCH',
				body: JSON.stringify({ model }),
			}),

		// Update LLM parameters (temperature, top_p, etc.)
		updateParameters: (params: UpdateLLMConfigRequest) =>
			fetchAPI<ModelConfig>('/api/models/parameters', {
				method: 'PATCH',
				body: JSON.stringify(params),
			}),
	},

	// Settings management
	settings: {
		// Get all settings
		getAll: () =>
			fetchAPI<SettingsResponse>('/api/settings'),

		// Get settings by category
		getCategory: (category: string) =>
			fetchAPI<
				| EmbeddingSettings
				| ChunkingSettings
				| RetrievalSettings
				| RerankerSettings
				| GenerationSettings
				| VectorStoreSettings
				| GPUSettings
			>(`/api/settings/${category}`),

		// Update settings (one or more categories)
		update: (settings: SettingsUpdateRequest) =>
			fetchAPI<SettingsUpdateResponse>('/api/settings', {
				method: 'PUT',
				body: JSON.stringify(settings),
			}),
	},

	// RAG queries
	rag: {
		// Semantic search query
		query: (params: QueryRequest) =>
			fetchAPI<QueryResponse>('/api/query', {
				method: 'POST',
				body: JSON.stringify(params),
			}),

		// RAG question answering
		ask: (params: RAGRequest) =>
			fetchAPI<RAGResponse>('/api/ask', {
				method: 'POST',
				body: JSON.stringify(params),
			}),
	},

	// Document ingestion
	ingestion: {
		// Ingest documents from path
		ingest: (params: IngestRequest) =>
			fetchAPI<IngestTaskResponse>('/api/ingest', {
				method: 'POST',
				body: JSON.stringify(params),
			}),

		// Check ingestion task status
		status: (taskId: string) =>
			fetchAPI<IngestionTask>(`/api/ingest/tasks/${taskId}`),

		// Clear knowledge base
		clear: (confirm = false) =>
			fetchAPI<ClearResponse>('/api/ingest/clear', {
				method: 'POST',
				body: JSON.stringify({ confirm }),
			}),

		// Get supported file formats
		formats: () =>
			fetchAPI<FormatsResponse>('/api/ingest/formats'),
	},

	// System health & info
	health: () => fetchAPI<{ status: string }>('/health'),
	
	status: () => fetchAPI<StatusResponse>('/api/status'),

	config: () => fetchAPI<Record<string, unknown>>('/api/config'),
};
