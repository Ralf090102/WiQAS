"""
Prompt Builder Module

This module defines classes and constants for constructing prompts used in WiQAS,
a RAG-driven Factoid Question Answering System specialized in Filipino culture.

Components:
    - FUNCTIONAL_GUIDELINES: Guidelines for shaping responses by type (Factual, Analytical, etc.).
    - EXEMPLARS: Few-shot examples demonstrating the expected QA style.
    - PromptTemplate: Encapsulates the logic for assembling system instructions,
      query, context, guidelines, and exemplars into a complete prompt.
    - PromptBuilder: Orchestrator that applies language detection (optional) and
      renders the final prompt via PromptTemplate.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from src.generation.query_classifier import QueryClassifier 

FUNCTIONAL_GUIDELINES = {
    "Factual": "Provide clear, concise, and accurate definitions, facts, or explanations. Focus on established knowledge and avoid unnecessary speculation. Use specific dates, names, and verifiable details when available.",
    "Analytical": "Offer thoughtful interpretation, highlight symbolism, make comparisons, and explain relationships or deeper meanings. Support reasoning with evidence or logical connections. Discuss cultural significance and historical context.",
    "Procedural": "Give structured, step-by-step instructions or processes. Ensure clarity, logical order, and completeness so the user can follow easily. Include materials needed and expected outcomes when relevant.",
    "Exploratory": "Deliver broad, contextual, and descriptive overviews. Provide background, key themes, and relevant connections without going too narrow or rigid. Explore multiple perspectives when appropriate.",
    "Comparative": "Compare and contrast different concepts, practices, traditions, or historical elements. Highlight similarities, differences, and cultural significance. Provide balanced analysis of each element being compared.",
}

EXEMPLARS = [
    # FACTUAL - FILIPINO
    {
        "question": "Kailan dineklara ang Batas Militar ni Ferdinand Marcos?",
        "context": "[1] On September 20, 1972, President Marcos went into prolonged conference with the armed forces high command. The proclamation was signed that evening, but Marcos apparently felt he needed one more incident to justify implementation. [Source: Filipino-Politics.pdf]\n\n[2] On the evening of September 23 the president explained his action on television. [Source: Filipino-Politics.pdf]",
        "answer": "Ang Batas Militar ay opisyal na idineklara ni Presidente Ferdinand Marcos noong Setyembre 23, 1972, sa pamamagitan ng television announcement, kahit na ang proklamasyon ay nilagdaan na noong Setyembre 20, 1972. [Source: Filipino-Politics.pdf]",
        "query_type": "Factual",
        "language": "fil"
    },
    {
        "question": "Ano ang paboritong kulay ni Emilio Aguinaldo?",
        "context": "No relevant documents found.",
        "answer": "Walang sapat na impormasyon sa mga dokumentong ito tungkol sa paboritong kulay ni Emilio Aguinaldo.",
        "query_type": "Factual",
        "language": "fil"
    },
    
    #FACTUAL - ENGLISH
    {
        "question": "What was the Laguna Copperplate Inscription?",
        "context": "[1] Discovered in 1989 buried in a riverbank in Laguna Province, south of Manila, the LCI was drawn up in 900 C.E., the equivalent of the Sanskrit calendar date inscribed on it—more than five hundred years before the barangay (village or settlement) of Maynila turned into a Muslim community. [Source: A-History-Of-The-Philippines ch1.pdf]\n\n[2] The inscription—the oldest known document of pre-Hispanic times in the Philippine archipelago—is vital to understanding how people in the islands, at least in certain parts, lived and what kind of society or societies they might have constructed. [Source: A-History-Of-The-Philippines ch1.pdf]",
        "answer": "The Laguna Copperplate Inscription (LCI) is the oldest known document from pre-Hispanic Philippines, dating to 900 C.E. Discovered in 1989 in Laguna Province, it provides crucial insights into pre-colonial Philippine society and contains a mix of Sanskrit, old Javanese, old Malay, and old Tagalog words. [Sources: A-History-Of-The-Philippines ch1.pdf]",
        "query_type": "Factual",
        "language": "eng"
    },
    {
        "question": "What was Jose Rizal's favorite food?",
        "context": "No relevant documents found.",
        "answer": "There is insufficient information in these documents about Jose Rizal's favorite food.",
        "query_type": "Factual",
        "language": "eng"
    },
    
    # ANALYTICAL - FILIPINO
    {
        "question": "Bakit mahalaga ang patron-client relationship sa pre-kolonyal na lipunan ng Pilipinas?",
        "context": "[1] The social relationship that tied leadership to the political economy in pre-Hispanic times was, as now, that of patron and client, a combination of cultural and functional linkages. [Source: Filipino-Politics.pdf]\n\n[2] The datu's land rights were meaningless without loyal cultivators, and so a leader's status and power came to be measured primarily in terms of the number of clients or followers he had. For the ordinary man the protection of a powerful patron was essential social security—and a psychological security as well. [Source: Filipino-Politics.pdf]",
        "answer": "Ang patron-client relationship ay sentral sa pre-kolonyal na lipunan dahil ito ay kumakatawan sa mutual dependency na nagbibigay ng social at psychological security. Para sa datu, walang saysay ang kanyang kapangyarihan at ari-arian kung walang mga tapat na tagasunod. Para naman sa karaniwang tao, ang proteksyon ng isang makapangyarihang patron ay essential social security. Ang sistemang ito ay sumasalamin din sa Filipino cultural values ng utang na loob at pakikipagkapwa-tao, kung saan ang relasyon ay hindi lamang transaksyonal kundi deeply rooted sa mutual obligations at reciprocity. [Source: Filipino-Politics.pdf]",
        "query_type": "Analytical",
        "language": "fil"
    },
    {
        "question": "Ano ang cultural significance ng bayanihan sa pre-kolonyal na Pilipinas?",
        "context": "No relevant documents found.",
        "answer": "Walang sapat na impormasyon sa mga dokumentong ito tungkol sa specific cultural significance ng bayanihan sa pre-kolonyal na Pilipinas.",
        "query_type": "Analytical",
        "language": "fil"
    },
    
    # ANALYTICAL - ENGLISH
    {
        "question": "How did the Spanish colonial period influence Filipino nationalism?",
        "context": "[1] The emerging nationalist was the first step in rebuilding the cultural self-confidence destroyed by the Spanish. It started with the acquisition of a much deeper understanding of European cultural roots, but Jose Rizal also addressed himself to the task of rediscovering the Malay heritage. [Source: Filipino-Politics.pdf]\n\n[2] True to Masonic influences, anti-clericalism became a persistent but not unalloyed strain in Filipino political philosophy. [Source: Filipino-Politics.pdf]\n\n[3] The ideological content of Filipino nationalism had also begun to take form during the Revolution. [Source: Filipino-Politics.pdf]",
        "answer": "The Spanish colonial period paradoxically both suppressed and catalyzed Filipino nationalism. While Spanish rule destroyed much of pre-colonial cultural confidence, it simultaneously created conditions for nationalist awakening. The ilustrados (educated elite) gained access to European Enlightenment ideas, which Jose Rizal and others used to critique colonial oppression while rediscovering Malay heritage. The execution of Filipino priests in 1872 became a turning point, transforming moderate reformism into revolutionary nationalism. Anti-clericalism emerged as a persistent theme due to friar abuses, though this targeted Spanish clergy rather than Filipino religious identity itself. The Revolution (1896-1899) marked the crystallization of Filipino national consciousness, blending European liberal ideals with indigenous values. [Source: Filipino-Politics.pdf]",
        "query_type": "Analytical",
        "language": "eng"
    },
    {
        "question": "What was the symbolic meaning of tattoos in pre-colonial Visayan society?",
        "context": "No relevant documents found.",
        "answer": "There is insufficient information in these documents about the symbolic meaning of tattoos in pre-colonial Visayan society.",
        "query_type": "Analytical",
        "language": "eng"
    },
    
    # PROCEDURAL - FILIPINO
    {
        "question": "Paano isinasagawa ang blood compact sa pre-kolonyal na Pilipinas?",
        "context": "[1] Alliances between two chiefs of different barangays would often be sealed through a blood compact, where each would drink wine mingled with the other's blood—or suck the blood directly from a cut made on the arm of the other. Simulating blood ties, the rite was called sandugo (or 'one blood'). [Source: A-History-Of-The-Philippines ch1.pdf]",
        "answer": "Ang blood compact o sandugo ay isinasagawa sa pamamagitan ng sumusunod na hakbang:\n\n1. Dalawang pinuno (chiefs) mula sa magkaibang barangay ay magtitipon para sa seremonya\n2. Isang hiwa o sugat ay ginagawa sa braso ng bawat pinuno\n3. Ang dugo mula sa hiwa ay pinagsasama sa alak (wine)\n4. Ang dalawang pinuno ay umiinom ng pinagsama nilang dugo sa alak, o direktang sinisipsip ang dugo mula sa sugat ng isa't isa\n5. Ang ritwal na ito ay tinatawag na 'sandugo' (one blood), na sumasalamin sa pagiging magkadugo na\n\nAng layunin ng blood compact ay mag-seal ng alliance at ipakita ang mutual trust at commitment sa pagitan ng dalawang barangay, na ginagawang symbolic brothers ang dalawang pinuno. [Source: A-History-Of-The-Philippines ch1.pdf]",
        "query_type": "Procedural",
        "language": "fil"
    },
    {
        "question": "Paano ginagawa ang traditional Filipino wedding ceremony sa pre-kolonyal na panahon?",
        "context": "No relevant documents found.",
        "answer": "Walang sapat na impormasyon sa mga dokumentong ito tungkol sa proseso ng traditional Filipino wedding ceremony sa pre-kolonyal na panahon.",
        "query_type": "Procedural",
        "language": "fil"
    },
    
    # PROCEDURAL - ENGLISH
    {
        "question": "How was the Constitutional Convention conducted during the Marcos era?",
        "context": "[1] By 1972 most observers concluded that Marcos planned to become prime minister under a new constitution that would not limit the tenure of the office. His liberal distribution of cash to delegates to the recently convened Constitutional Convention (the 'Con Con') did secure endorsement of the parliamentary system. [Source: Filipino-Politics.pdf]\n\n[2] But while Marcos could, with envelopes of peso bills, influence the vote, he found it more difficult to get the delegates to conclude their work. By August 1972 some delegates were openly discussing the advisability of adjourning until after the 1973 election. [Source: Filipino-Politics.pdf]",
        "answer": "The Constitutional Convention (Con Con) during the Marcos era proceeded through the following process:\n\n1. **Delegate Selection**: Delegates were elected to draft a new constitution\n2. **Convening**: The Constitutional Convention was formally convened with delegates from various provinces\n3. **Deliberations**: Delegates debated various constitutional provisions, including the shift to a parliamentary system\n4. **Influence and Lobbying**: President Marcos actively influenced proceedings by:\n   - Distributing cash payments to delegates (peso bills in envelopes)\n   - Securing votes for his preferred provisions\n5. **Challenges**: Despite financial influence, Marcos faced difficulties getting delegates to complete their work quickly\n6. **延期 Discussions**: By August 1972, some delegates proposed adjourning until after the 1973 election to \"avoid politics\"\n\nThe convention's progress was ultimately interrupted by the declaration of Martial Law on September 21, 1972, which altered the constitutional transition process. [Source: Filipino-Politics.pdf]",
        "query_type": "Procedural",
        "language": "eng"
    },
    {
        "question": "How were babaylans chosen in pre-colonial Filipino communities?",
        "context": "No relevant documents found.",
        "answer": "There is insufficient information in these documents about the selection process for babaylans in pre-colonial Filipino communities.",
        "query_type": "Procedural",
        "language": "eng"
    },
    
    # EXPLORATORY - FILIPINO
    {
        "question": "Ano ang katangian ng lipunan sa pre-kolonyal na Pilipinas?",
        "context": "[1] The smallest politico-social unit was the barangay, which, as has been noted, originally referred to the seagoing vessel on which a family or clan traveled. [Source: A-History-Of-The-Philippines ch1.pdf]\n\n[2] A datu headed the barangay, with each barangay more or less self-contained, though there were occasionally alliances among various barangays. [Source: A-History-Of-The-Philippines ch1.pdf]\n\n[3] Native societies, as the Spanish found them, were, like Caesar's Gaul, divided into three parts: the ruling elite, their peers and followers, and slaves. [Source: A-History-Of-The-Philippines ch1.pdf]",
        "answer": "Ang pre-kolonyal na lipunan ng Pilipinas ay binubuo ng iba't ibang katangian:\n\n**Pampulitikang Istruktura**: Ang barangay ang pinakamaliit na yunit, na pinamumunuan ng isang datu. Bawat barangay ay self-contained ngunit may mga alliances sa iba pang barangay.\n\n**Social Hierarchy**: Tatlong pangunahing grupo ang bumubuo ng lipunan:\n- Ang ruling elite (mga datu at maginoo)\n- Ang kanilang mga followers (timawa at maharlika)\n- Ang mga alipin (may dalawang uri: namamahay at sagigilid)\n\n**Ekonomiya**: Nakabase sa komunal land use, shared labor (bayanihan), at patron-client relationships. Walang konsepto ng private land ownership.\n\n**Kultura**: Bilateral kinship system, gender equality, animistic beliefs na may Hindu at Buddhist influences, at maritime trading culture.\n\nAng sistemang ito ay sumasalamin sa Filipino values ng pakikipagkapwa-tao, bayanihan, at utang na loob na patuloy na bahagi ng kultura hanggang ngayon. [Sources: A-History-Of-The-Philippines ch1.pdf]",
        "query_type": "Exploratory",
        "language": "fil"
    },
    {
        "question": "Ano ang papel ng kababaihan sa pre-kolonyal na ekonomiya?",
        "context": "No relevant documents found.",
        "answer": "Walang sapat na impormasyon sa mga dokumentong ito tungkol sa specific na papel ng kababaihan sa pre-kolonyal na ekonomiya ng Pilipinas.",
        "query_type": "Exploratory",
        "language": "fil"
    },
    
    # EXPLORATORY - ENGLISH
    {
        "question": "What was the American colonial experience like in the Philippines?",
        "context": "[1] The American period, despite its brevity and because of its recency, has left a political legacy in the contemporary Philippines perhaps as great as that of the Spanish era. [Source: Filipino-Politics.pdf]\n\n[2] The expansion of political participation was perhaps the greatest change. In 1907 the first elected legislature in Southeast Asia was chosen by an electorate limited by property qualifications. [Source: Filipino-Politics.pdf]\n\n[3] Education had a relatively high priority during the American period. The school population expanded 500 percent in a generation, and educational expenditures came to consume one-half of government budgets at all levels. [Source: Filipino-Politics.pdf]",
        "answer": "The American colonial experience (1898-1946) profoundly shaped modern Philippine society through several key aspects:\n\n**Political Development**: Americans introduced democratic institutions including the first elected legislature in Southeast Asia (1907), gradually expanding suffrage from property-qualified voters to all literate males by 1916. This created a tradition of electoral politics that persists today.\n\n**Education Revolution**: Educational expansion was dramatic—school population grew 500% within a generation, with education consuming half of government budgets. English became the language of instruction, creating widespread bilingualism. By the 1930s, literacy nearly doubled to 50%.\n\n**Economic Ties**: The Payne-Aldrich Tariff (1909) established free trade, integrating Philippine economy with the US market while limiting local manufacturing. This created a colonial economic relationship that privileged export agriculture.\n\n**Cultural Impact**: American popular culture (movies, music, fashion) became deeply embedded. The period reinforced Western orientation while paradoxically enabling nationalist movements through education and political participation.\n\n**Ambivalent Legacy**: While Americans promoted self-governance and education, they also perpetuated colonial relationships and failed to address fundamental social inequalities, particularly land reform. [Source: Filipino-Politics.pdf]",
        "query_type": "Exploratory",
        "language": "eng"
    },
    {
        "question": "What were the traditional Filipino concepts of the afterlife?",
        "context": "No relevant documents found.",
        "answer": "There is insufficient information in these documents about traditional Filipino concepts of the afterlife in pre-colonial times.",
        "query_type": "Exploratory",
        "language": "eng"
    },
    
    # COMPARATIVE - FILIPINO
    {
        "question": "Ano ang pagkakaiba ng patron-client system sa pre-kolonyal at kolonyal na panahon?",
        "context": "[1] The social relationship that tied leadership to the political economy in pre-Hispanic times was, as now, that of patron and client, a combination of cultural and functional linkages. [Source: Filipino-Politics.pdf]\n\n[2] But in the bargain between patron and client benefits flowed both ways—not the stuff of which absolutism is easily made. Mutuality, plus the tendency toward consultative decision making among barangay elders, have led some Filipinos to regard the barangay as the cradle of Filipino democracy. [Source: Filipino-Politics.pdf]\n\n[3] The party system had merely strengthened the political hold of dominant, landed families, which had used quasi-feudal techniques to gain an electoral following and could hold national power through a system of one-party dominance. [Source: Filipino-Politics.pdf]",
        "answer": "Ang patron-client system ay nag-evolve nang malaki mula pre-kolonyal hanggang kolonyal na panahon:\n\n**Pre-Kolonyal na Sistema**:\n- **Mutuality**: Ang benefits ay dumaloy sa dalawang direksyon—hindi one-sided ang relasyon\n- **Consultative**: May tendency ng collaborative decision-making kasama ang barangay elders\n- **Social Security**: Ang protection ng patron ay essential para sa psychological at social security ng client\n- **Flexibility**: May social mobility—posible ang pag-ahon mula sa status ng alipin\n- **Democratic Elements**: Ang barangay ay nakikita bilang \"cradle of Filipino democracy\" dahil sa mutual obligations\n\n**Kolonyal at Post-Kolonyal na Sistema**:\n- **Quasi-feudal**: Naging mas rigid at hierarchical ang structure\n- **Electoral Manipulation**: Ginamit ang patron-client ties para kontrolin ang boto at electoral outcomes\n- **Land-based Power**: Ang landed families ay nag-monopolize ng political power\n- **One-party Dominance**: Ang sistema ay nag-entrench ng political dynasties\n- **Reduced Mutuality**: Ang reciprocity ay naging mas transactional at less balanced\n\nAng pangunahing pagkakaiba ay nasa shift from a relatively fluid, mutual system tungo sa isang mas exploitative at institutionalized na hierarchy na ginamit para sa electoral at economic control. [Sources: Filipino-Politics.pdf]",
        "query_type": "Comparative",
        "language": "fil"
    },
    {
        "question": "Ano ang pagkakaiba ng mga babaylan at mga pari noong Spanish colonial period?",
        "context": "No relevant documents found.",
        "answer": "Walang sapat na impormasyon sa mga dokumentong ito para magbigay ng comparative analysis ng babaylan at mga paring Espanyol noong kolonyal na panahon.",
        "query_type": "Comparative",
        "language": "fil"
    },
    
    # COMPARATIVE - ENGLISH 
    {
        "question": "How did Spanish and American colonial education systems differ in the Philippines?",
        "context": "[1] Iberian Catholicism, which completely dominated education, taught by precept and example that the church itself had the right to play a major, if not controlling, role in social, political, and economic life. [Source: Filipino-Politics.pdf]\n\n[2] Natives were discouraged from learning Spanish, thus giving the bilingual friars a crucial role. [Source: Filipino-Politics.pdf]\n\n[3] Education had a relatively high priority during the American period. The school population expanded 500 percent in a generation, and educational expenditures came to consume one-half of government budgets at all levels. Indeed, educational opportunity in the Philippines was greater than in any other colony in Asia. [Source: Filipino-Politics.pdf]\n\n[4] Literacy doubled to nearly 50 percent in the 1930s and English, the language of instruction, was spoken by 27 percent of the population. [Source: Filipino-Politics.pdf]",
        "answer": "Spanish and American colonial education systems in the Philippines differed fundamentally in philosophy, accessibility, and impact:\n\n**Spanish Colonial Education (1565-1898)**:\n- **Religious Dominance**: Catholic Church completely controlled education\n- **Limited Access**: Primarily for elite ilustrados; masses received minimal formal education\n- **Language Policy**: Natives were actively discouraged from learning Spanish, maintaining friar intermediary power\n- **Purpose**: Indoctrination into Catholicism and maintaining colonial hierarchy\n- **Literacy**: Low rates; education served to preserve social stratification\n\n**American Colonial Education (1898-1946)**:\n- **Secular Emphasis**: Though influenced by Protestant values, focused on civic and practical education\n- **Mass Accessibility**: 500% expansion in school population; highest educational opportunity among Asian colonies\n- **Language Policy**: English as medium of instruction, deliberately creating bilingual population (27% English proficiency by 1930s)\n- **Budget Priority**: Education consumed 50% of government budgets at all levels\n- **Democratic Intent**: Aimed at creating informed citizenry capable of self-governance\n- **Literacy Impact**: Literacy doubled to 50% by the 1930s\n\n**Key Contrast**: Spanish education maintained colonial control through exclusion and religious monopoly, while American education promoted broader participation—though still serving colonial interests through cultural assimilation and economic integration. [Sources: Filipino-Politics.pdf]",
        "query_type": "Comparative",
        "language": "eng"
    },
    {
        "question": "What were the differences between Tagalog and Visayan pre-colonial societies?",
        "context": "No relevant documents found.",
        "answer": "There is insufficient information in these documents to provide a detailed comparison of Tagalog and Visayan pre-colonial societies.",
        "query_type": "Comparative",
        "language": "eng"
    }
]

class PromptTemplate:
    """
    Defines the hierarchical structure for prompt construction.

    Template Sections:
        - System Instructions: Core principles guiding the model's behavior.
        - Context: Retrieved snippets formatted as bullet points.
        - Query: The user's question.
        - Guidelines: Response style instructions based on query type.
        - Exemplars: Few-shot examples illustrating the desired QA style.
    """

    def __init__(
        self,
        query: str,
        context: list[str] | list[dict[str, Any]],
        query_type: str = "Factual",
        language: str = "fil",
        include_exemplars: bool = True,
        use_detailed_context: bool = True,
    ):
        """
        Initialize a PromptTemplate instance.

        Args:
            query (str): User question.
            context (List[str]): Retrieved context snippets.
            query_type (str, optional): Response style guideline (default: "Factual").
            language (str, optional): Language code (default: "fil").
        """
        self.query = query
        self.context = context
        self.query_type = query_type
        self.language = language
        self.include_exemplars = include_exemplars
        self.use_detailed_context = use_detailed_context

    def _format_source_citation(self, ctx: dict[str, Any]) -> str:
        citation_text = ctx.get("citation_text")
        if citation_text:
            return f"[Source: {citation_text}]"

        source_file = ctx.get("normalized_source_file") or ctx.get("source_file", "")
        if source_file:
            return f"[Source: {source_file}]"

        return "[Source: Unknown]"

    def build_system_instructions(self) -> str:
        """
        Construct the system instructions section of the prompt.

        Returns:
            str: Instruction string containing principles of WiQAS such as
            factual accuracy, cultural faithfulness, and citation requirements.
        """

        citation_examples = (
            "Citation Format Examples:\n"
            "- For PDFs: [Source: Food Of The Philippines, p. 23] \n"
            "- For Wikipedia: [Source: Article Title (Wikipedia, accessed January 15, 2024)]\n"
            '- For News: [Source: "Article Title", January 15, 2024. Retrieved from URL]\n'
            "- For Books: [Source: Book Title, p. 45]\n"
            '- Multiple sources: [Sources: Food Of The Philippines, p. 23; Article Title (Wikipedia, accessed January 15, 2024); "Article Title", January 15, 2024. Retrieved from URL]\n'
            "- No sources: do not put any citations\n\n"
        )

        return (
            "You are WiQAS, a RAG-driven Factoid Question Answering System specialized in Filipino culture. "
            "Your role is to generate answers grounded in the retrieved context from the knowledge base. "
            "Follow these principles:\n\n"

            "1. **Factual Accuracy**: Only use information found in the provided context. If the context "
            "does not contain enough information, clearly state: 'Walang sapat na impormasyon sa mga "
            "dokumentong ito tungkol sa [topic]' (Filipino) or 'There is insufficient information in "
            "these documents about [topic]' (English).\n\n"

            "2. **Cultural Faithfulness**: Ensure responses respect Filipino linguistic, historical, and "
            "cultural nuances. Preserve cultural authenticity when explaining concepts, practices, or traditions. "
            "Recognize the diversity within Filipino culture across regions and time periods.\n\n"

            "3. **Clarity & Precision**: Provide concise, factoid-style answers unless the question calls "
            "for elaboration. Avoid unnecessary speculation or overgeneralization. Use specific details like "
            "dates, names, and locations when available in the context.\n\n"

            "4. **Context-Aware Language**: Pay attention to Filipino semantic and linguistic nuances, including "
            "natural code-switching between Filipino and English as commonly practiced in Philippine discourse. "
            "When relevant, explain terms, transliterations, or culturally-specific phrases. Match the primary "
            "language of your response to the user's query language, but use the most appropriate language for "
            "specific terms and concepts. Provide translations or explanations when using terms that might not "
            "be familiar to the target audience.\n\n"

            f"5. **Detailed Citations**: Always reference sources at the end of your answer using the exact citation in the citation "
            "format provided in the context snippets unless there are unsufficient information for a response. Each context snippet includes a [Source: ...] citation - "
            "use these citations directly in your answer. When multiple sources support your answer, list all of them. "
            "Citations must be accurate, detailed, and traceable. The generated answer should always end with the citation, unless No relevant documents found.\n\n"
            f"{citation_examples}"

            "6. **Handling Uncertainty**: If the context provides partial information, answer what you can and "
            "explicitly state what information is missing. Never fabricate details not present in the context.\n\n"

            "Remember: WiQAS is not a generic QA system—it is designed specifically to answer questions "
            "about Filipino culture accurately, faithfully, and in context. Your responses should demonstrate "
            "cultural competence and linguistic awareness appropriate for Filipino cultural topics."
        )

    def build_context_section(self) -> str:
        """
        Construct the context section of the prompt.

        Returns:
            str: Bullet-pointed list of context snippets or a fallback message.
        """
        if not self.context:
            return "No relevant documents found."

        formatted_contexts = []

        for i, ctx in enumerate(self.context, 1):
            if isinstance(ctx, dict):
                text = ctx.get("text", "")

                if self.use_detailed_context:
                    citation = self._format_source_citation(ctx)
                    formatted_contexts.append(f"[{i}] {text} {citation}")
                else:
                    formatted_contexts.append(f"[{i}] {text}")
            else:
                formatted_contexts.append(f"[{i}] {ctx}")

        return "\n\n".join(formatted_contexts)

    def build_query_section(self) -> str:
        """
        Construct the query section of the prompt.

        Returns:
            str: Formatted user question.
        """
        return f"User Question:\n{self.query}\n\n" f"[Detected Language: {self.language.upper()} | Query Type: {self.query_type}]"

    def build_guidelines(self) -> str:
        """
        Construct the response guidelines section of the prompt.

        Returns:
            str: Guidelines text derived from FUNCTIONAL_GUIDELINES and
            contextualized for the query type.
        """
        guideline = FUNCTIONAL_GUIDELINES.get(self.query_type)

        # Enhanced multilingual instructions
        if self.language == "fil":
            language_instruction = (
                "Respond primarily in Filipino (Tagalog), but use code-switching with English when:\n"
                "  • Technical terms are more commonly known in English\n"
                "  • Proper nouns or specific terminology from the source context is in English\n"
                "  • The context contains mixed language content that's better preserved as-is"
            )
        else:
            language_instruction = (
                "Respond in English, but include Filipino terms when:\n"
                "  • The Filipino term is culturally significant or has no direct English equivalent\n"
                "  • Proper nouns or cultural concepts are better understood in Filipino\n"
                "  • Direct quotes from Filipino sources should be preserved"
            )

        return (
            f"Response Guidelines ({self.query_type} Type):\n"
            f"{guideline}\n\n"
            f"Additional Requirements:\n"
            f"- {language_instruction}\n"
            f"- Maintain cultural sensitivity and authenticity\n"
            f"- If information is insufficient, state this clearly rather than speculating\n"
            f"- When relevant, provide cultural context and explanations for Filipino-specific terms\n"
        )

    def build_exemplars(self) -> str:
        """
        Construct the few-shot exemplar section of the prompt.

        Returns:
            str: Example question-context-answer triplets formatted for prompting.
        """
        if not self.include_exemplars:
            return ""

        relevant_exemplars = [ex for ex in EXEMPLARS if ex.get("language", "fil") == self.language or ex.get("query_type", "") == self.query_type]

        if not relevant_exemplars:
            relevant_exemplars = EXEMPLARS

        exemplars_text = []
        for ex in relevant_exemplars[:3]:
            exemplars_text.append(f"Example Question: {ex['question']}\n" f"Context Provided:\n{ex['context']}\n" f"Expected Answer: {ex['answer']}")
        return "\n\n".join(exemplars_text)

    def render(self) -> str:
        """
        Render the complete prompt by combining all sections.

        Returns:
            str: Fully constructed prompt string with system instructions,
            context, query, guidelines, and exemplars.
        """
        sections = [
            f"=== SYSTEM INSTRUCTIONS ===\n{self.build_system_instructions()}",
            f"=== RETRIEVED CONTEXT ===\n{self.build_context_section()}",
            f"=== USER QUERY ===\n{self.build_query_section()}",
            f"=== RESPONSE GUIDELINES ===\n{self.build_guidelines()}",
        ]

        if self.include_exemplars:
            sections.append(f"=== FEW-SHOT EXAMPLES ===\n{self.build_exemplars()}")

        sections.append("=== YOUR ANSWER ===")

        return "\n\n" + ("-" * 80 + "\n\n").join(sections)


class PromptBuilder:
    """
    Orchestrates construction of prompts with support for language detection.

    Attributes:
        detect_language_fn (Callable, optional): Function to infer language from query.
    """

    def __init__(self, detect_language_fn: Callable | None = None, use_classifier: bool = True, use_detailed_context: bool = True):
        """
        Initialize a PromptBuilder.

        Args:
            detect_language_fn (Callable, optional): Function that infers query language.
        """
        self.detect_language_fn = detect_language_fn
        self.use_classifier = use_classifier
        self.use_detailed_context = use_detailed_context
        self.classifier = QueryClassifier() if use_classifier else None

    def build_prompt(
        self,
        query: str,
        context: list[str] | list[dict[str, Any]],
        query_type: str | None = None,
        language: str | None = None,
        include_exemplars: bool = True,
    ) -> str:
        """
        Build the final prompt using PromptTemplate.

        Args:
            query (str): User question.
            context (List[str]): Retrieved context snippets.
            query_type (str, optional): Desired response style (default: "Factual").
            language (str, optional): Target response language (default: inferred or "fil").

        Returns:
            str: Fully constructed prompt string.
        """
        if self.use_classifier and self.classifier:
            classification = self.classifier.classify(query)

            if query_type is None:
                query_type = classification.query_type
            if language is None:
                language = classification.language

        if self.detect_language_fn and language is None:
            language = self.detect_language_fn(query)

        query_type = query_type or "Factual"
        language = language or "fil"

        template = PromptTemplate(
            query=query,
            context=context,
            query_type=query_type,
            language=language,
            include_exemplars=include_exemplars,
            use_detailed_context=self.use_detailed_context,
        )
        return template.render()
