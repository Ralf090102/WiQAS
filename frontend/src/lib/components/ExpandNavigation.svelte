<script lang="ts">
	interface Props {
		isCollapsed: boolean;
		onClick: () => void;
		classNames: string;
	}

	let { isCollapsed, classNames, onClick }: Props = $props();
</script>

<button
	onclick={onClick}
	title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
	class="{classNames} group flex h-16 w-6 flex-col items-center justify-center -space-y-1 outline-none *:h-3 *:w-1 *:rounded-full transition-all hover:scale-110 {!isCollapsed
		? '*:bg-blue-400 dark:*:bg-blue-600 *:hover:bg-blue-600 dark:*:hover:bg-blue-400'
		: '*:bg-blue-400 dark:*:bg-blue-600 *:hover:bg-blue-600 dark:*:hover:bg-blue-400'}"
	name="sidebar-toggle"
	aria-label="Toggle sidebar navigation"
>
	<div class={!isCollapsed ? "group-hover:rotate-[20deg] transition-transform" : "group-hover:-rotate-[20deg] transition-transform"}></div>
	<div class={!isCollapsed ? "group-hover:-rotate-[20deg] transition-transform" : "group-hover:rotate-[20deg] transition-transform"}></div>
</button>
