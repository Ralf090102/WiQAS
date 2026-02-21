import { UrlDependency } from "$lib/types/UrlDependency";
import type { ConvSidebar } from "$lib/types/ConvSidebar";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

export const load = async ({ depends, fetch }) => {
	depends(UrlDependency.ConversationList);

	// Chat sessions are out of scope - WiQAS is a pure QA system with no conversation history
	const conversations: ConvSidebar[] = [];

	// Fetch actual active model from backend
	let activeModelName = 'mistral:latest';
	try {
		const response = await fetch(`${BACKEND_URL}/api/models/config`);
		if (response.ok) {
			const config = await response.json();
			activeModelName = config.model;
		}
	} catch (err) {
		console.error('Failed to load model config:', err);
	}

	// Create model entry based on active model
	const models = [
		{
			id: activeModelName,
			name: activeModelName,
			displayName: activeModelName,
			description: 'Ollama model',
			websiteUrl: '',
			modelUrl: '',
			datasetName: '',
			datasetUrl: '',
			preprompt: '',
			chatPromptTemplate: '',
			parameters: {
				temperature: 0.7,
				top_p: 0.95,
				max_new_tokens: 2048,
			},
		},
	];

	// Mock public config
	const publicConfig = {
		PUBLIC_APP_NAME: import.meta.env.PUBLIC_APP_NAME || 'WiQAS',
		PUBLIC_APP_DESCRIPTION: import.meta.env.PUBLIC_APP_DESCRIPTION || 'Local RAG Chat',
		PUBLIC_ORIGIN: import.meta.env.PUBLIC_ORIGIN || 'http://34.124.143.216:3000',
		isHuggingChat: false,
		assetPath: '/chatui',
		PUBLIC_PLAUSIBLE_SCRIPT_URL: undefined,
		PUBLIC_APPLE_APP_ID: undefined,
	};

	// Settings with actual active model
	const settings = {
		activeModel: activeModelName,
		customPrompts: {},
		hidePromptExamples: {},
		multimodalOverrides: {},
		toolsOverrides: {},
		welcomeModalSeen: true,
		welcomeModalSeenAt: null,
		directPaste: false,
		disableStream: false,
		shareConversationsWithModelAuthors: false,
	};

	return {
		conversations,
		models,
		oldModels: [],
		user: null, // No authentication for local use
		settings,
		publicConfig,
		transcriptionEnabled: true, // Enable microphone button
		tools: [],
	};
};
