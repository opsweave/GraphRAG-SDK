import logging
from falkordb import Graph
from typing import Optional
from graphrag_sdk.steps.Step import Step
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.models import (
    GenerativeModelChatSession,
)
from graphrag_sdk.helpers import (
    check_falkordb_result_set_empty,
    extract_cypher,
    validate_cypher,
    stringify_falkordb_response,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GraphQueryGenerationStep(Step):
    """
    Graph Query Step
    """

    def __init__(
        self,
        graph: Graph,
        ontology: Ontology,
        chat_session: GenerativeModelChatSession,
        config: Optional[dict] = None,
        last_answer: Optional[str] = None,
        cypher_prompt: Optional[str] = None,
        cypher_prompt_with_history: Optional[str] = None,
    ) -> None:
        """
        Initializes the GraphQueryGenerationStep object.
        
        Args:
            graph (Graph): The graph object to query.
            ontology (Ontology): The ontology object.
            chat_session (GenerativeModelChatSession): The chat session object.
            config (Optional[dict]): The configuration object.
            last_answer (Optional[str]): The last answer.
            cypher_prompt (Optional[str]): The Cypher prompt.
            cypher_prompt_with_history (Optional[str]): The Cypher prompt with history.
        """
        self.ontology = ontology
        self.config = config or {}
        self.graph = graph
        self.chat_session = chat_session
        self.last_answer = last_answer
        self.cypher_prompt = cypher_prompt
        self.cypher_prompt_with_history = cypher_prompt_with_history

    def run(self, question: str, retries: Optional[int] = 10) -> tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
        """
        Run the step to generate and validate a Cypher query.
        
        Args:
            question (str): The question being asked to generate the query.
            retries (Optional[int]): Number of retries allowed in case of errors.
            
        Returns:
            tuple[Optional[str], Optional[str], Optional[int], Optional[str]]: The context, 
            the generated Cypher query and the query execution time in milliseconds, and the display Cypher query.
        """
        cypher = ""
        cypher_prompt = ""
        last_empty_query = None
        for i in range(retries):
            try:
                if cypher_prompt == "":
                    cypher_prompt = (
                        (self.cypher_prompt.format(question=question) 
                        if self.last_answer is None
                        else self.cypher_prompt_with_history.format(question=question, last_answer=self.last_answer))
                    )
                logger.debug(f"Cypher Prompt: {cypher_prompt}")
                cypher_statement_response = self.chat_session.send_message(
                    cypher_prompt,
                )
                logger.debug(f"Cypher Statement Response: {cypher_statement_response}")
                result = extract_cypher(cypher_statement_response.text)
                logger.debug(f"Cypher: {result}")

                if not result or "error" in result:
                    return (None, None, None, None)
                
                if "context" not in result or result["context"] is None:
                    return (None, None, None, None)

                cypher = result["context"]

                validation_errors = validate_cypher(cypher, self.ontology)
                if validation_errors is not None:
                    raise Exception("\n".join(validation_errors))

                if cypher is not None:
                    query_result = self.graph.query(cypher)
                    result_set = query_result.result_set
                    execution_time = query_result.run_time_ms

                    is_empty_result = check_falkordb_result_set_empty(result_set)
                    if is_empty_result and cypher != last_empty_query:
                        logger.debug("Cypher query returned empty result set, retrying...")
                        last_empty_query = cypher
                        cypher_prompt = f"The generated Cypher query returned an empty result set. If you are absolutely sure there is another way to formulate the query based on the ontology to answer the question, please try again. Otherwise, return the same Cypher query. The question to answer is: {question}"
                        continue
                    
                    context = stringify_falkordb_response(result_set)
                    logger.debug(f"Context: {context}")
                    logger.debug(f"Context size: {len(result_set)}")
                    logger.debug(f"Context characters: {len(str(context))}")

                    return (context, cypher, execution_time, result["display"] if "display" in result else None)
            except Exception as e:
                logger.error(f"Error: {e}")
                error = e
                if "litellm.RateLimitError" in str(e):
                    self.chat_session.delete_last_message()
                    continue
                cypher_prompt = f"The previous Cypher query you generated resulted in the following error: {str(e)}. Please generate a new Cypher query that adheres to the ontology and avoids this error. The question to answer is: {question}"

        raise Exception("Failed to generate Cypher query: " + str(error))
