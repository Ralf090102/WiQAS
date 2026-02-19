<script lang="ts">
	import { base } from "$app/paths";
	import { page } from "$app/state";

	import CarbonTrashCan from "~icons/carbon/trash-can";
	import CarbonEdit from "~icons/carbon/edit";
	import type { ConvSidebar } from "$lib/types/ConvSidebar";

	import EditConversationModal from "$lib/components/EditConversationModal.svelte";
	import DeleteConversationModal from "$lib/components/DeleteConversationModal.svelte";
	import { requireAuthUser } from "$lib/utils/auth";

	interface Props {
		conv: ConvSidebar;
		readOnly?: true;
		ondeleteConversation?: (id: string) => void;
		oneditConversationTitle?: (payload: { id: string; title: string }) => void;
	}

	let { conv, readOnly, ondeleteConversation, oneditConversationTitle }: Props = $props();

	let deleteOpen = $state(false);
	let renameOpen = $state(false);
</script>

<a
	data-sveltekit-noscroll
	data-sveltekit-preload-data="tap"
	href="{base}/conversation/{conv.id}"
	class="group flex h-10 flex-none items-center gap-2 rounded-xl border px-3 text-gray-700 transition-all hover:scale-[1.02] hover:shadow-md dark:text-gray-300
		{conv.id === page.params.id 
			? 'border-blue-300 bg-gradient-to-r from-blue-100 to-indigo-100 shadow-sm dark:border-blue-700 dark:from-blue-950/80 dark:to-indigo-950/80' 
			: 'border-gray-200 bg-white hover:border-blue-200 hover:bg-blue-50/50 dark:border-gray-700 dark:bg-gray-800/70 dark:hover:border-blue-800 dark:hover:bg-gray-700'}"
>
	<div class="min-w-0 flex-1 truncate first-letter:uppercase">
		<span class="text-sm font-medium {conv.id === page.params.id ? 'text-blue-700 dark:text-blue-300' : ''}">{conv.title}</span>
	</div>

	{#if !readOnly}
		<button
			type="button"
			class="flex size-7 items-center justify-center rounded-lg transition-all md:opacity-0 md:group-hover:opacity-100 hover:bg-blue-100 hover:scale-110 dark:hover:bg-blue-900/50"
			title="Edit conversation title"
			onclick={(e) => {
				e.preventDefault();
				if (requireAuthUser()) return;
				renameOpen = true;
			}}
		>
			<CarbonEdit class="text-sm text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-300" />
		</button>

		<button
			type="button"
			class="flex size-7 items-center justify-center rounded-lg transition-all md:opacity-0 md:group-hover:opacity-100 hover:bg-red-100 hover:scale-110 dark:hover:bg-red-900/40"
			title="Delete conversation"
			onclick={(event) => {
				event.preventDefault();
				if (requireAuthUser()) return;
				if (event.shiftKey) {
					ondeleteConversation?.(conv.id.toString());
				} else {
					deleteOpen = true;
				}
			}}
		>
			<CarbonTrashCan class="text-sm text-gray-500 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400" />
		</button>
	{/if}
</a>

<!-- Edit title modal -->
{#if renameOpen}
	<EditConversationModal
		open={renameOpen}
		title={conv.title}
		onclose={() => (renameOpen = false)}
		onsave={(payload) => {
			renameOpen = false;
			oneditConversationTitle?.({ id: conv.id.toString(), title: payload.title });
		}}
	/>
{/if}

<!-- Delete confirmation modal -->
{#if deleteOpen}
	<DeleteConversationModal
		open={deleteOpen}
		title={conv.title}
		onclose={() => (deleteOpen = false)}
		ondelete={() => {
			deleteOpen = false;
			ondeleteConversation?.(conv.id.toString());
		}}
	/>
{/if}
