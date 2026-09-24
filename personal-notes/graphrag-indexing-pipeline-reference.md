# GraphRAG Indexing Pipeline — Comprehensive Reference

> **Purpose:** Detailed technical documentation of the GraphRAG indexing pipeline for creating a Copilot custom agent.
> **Based on:** GraphRAG v3.2.0
> **Date:** September 2026

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Pipeline Modes](#pipeline-modes)
3. [Entry Points & Orchestration](#entry-points--orchestration)
4. [Common Infrastructure](#common-infrastructure)
5. [Step 1: Load Input Documents](#step-1-load-input-documents)
6. [Step 2: Create Base Text Units (Chunking)](#step-2-create-base-text-units-chunking)
7. [Step 3: Create Final Documents](#step-3-create-final-documents)
8. [Step 4: Extract Entity Graph](#step-4-extract-entity-graph)
9. [Step 4 (NLP Variant): Extract Graph via NLP](#step-4-nlp-variant-extract-graph-via-nlp)
10. [Step 5: Prune Graph (Fast mode only)](#step-5-prune-graph-fast-mode-only)
11. [Step 6: Finalize Graph](#step-6-finalize-graph)
12. [Step 7: Extract Covariates (Claims)](#step-7-extract-covariates-claims)
13. [Step 8: Create Communities](#step-8-create-communities)
14. [Step 9: Create Final Text Units](#step-9-create-final-text-units)
15. [Step 10: Create Community Reports](#step-10-create-community-reports)
16. [Step 11: Generate Text Embeddings](#step-11-generate-text-embeddings)
17. [Incremental Update Pipeline](#incremental-update-pipeline)
18. [Output Data Schema](#output-data-schema)
19. [Configuration Reference](#configuration-reference)
20. [End-to-End Data Flow Diagram](#end-to-end-data-flow-diagram)

---

## Pipeline Overview

The GraphRAG indexing pipeline transforms **unstructured text documents** into a **structured knowledge graph** with:
- Entities (nodes)
- Relationships (edges)
- Covariates (claims extracted from text)
- Community hierarchy (clusters of related entities)
- Community reports (LLM-generated summaries of each community)
- Text embeddings (vector representations for semantic search)

### Two Indexing Modes

| Mode | Graph Extraction Method | Workflows | Best For |
|---|---|---|---|
| **Standard** | LLM-based entity/relationship extraction | 10 workflows | Accuracy, rich entity extraction |
| **Fast** | NLP-based noun phrase extraction | 10 workflows | Speed, lower cost |

### Two Run Types

| Type | Description |
|---|---|
| **Standard Run** | Full indexing from scratch |
| **Update Run** | Incremental indexing — processes only new/changed documents and merges with existing index |
## Pipeline Modes

### Standard Pipeline (IndexingMethod.Standard)

```
[load_input_documents] → [create_base_text_units] → [create_final_documents] →
[extract_graph] → [finalize_graph] → [extract_covariates] → [create_communities] →
[create_final_text_units] → [create_community_reports] → [generate_text_embeddings]
```

**Workflow registration** (from `packages/graphrag/graphrag/index/workflows/factory.py`):
```python
_standard_workflows = [
    "create_base_text_units", "create_final_documents",
    "extract_graph", "finalize_graph", "extract_covariates",
    "create_communities", "create_final_text_units",
    "create_community_reports", "generate_text_embeddings",
]
PipelineFactory.register_pipeline(
    IndexingMethod.Standard, ["load_input_documents", *_standard_workflows]
)
```

### Fast Pipeline (IndexingMethod.Fast)

```
[load_input_documents] → [create_base_text_units] → [create_final_documents] →
[extract_graph_nlp] → [prune_graph] → [finalize_graph] → [create_communities] →
[create_final_text_units] → [create_community_reports_text] → [generate_text_embeddings]
```

**Differences from Standard:**
- Uses `extract_graph_nlp` (noun phrase extraction) instead of `extract_graph` (LLM extraction)
- Adds `prune_graph` step to clean up noise from NLP extraction
- Uses `create_community_reports_text` (text-based reports) instead of `create_community_reports` (graph-based reports)

### Update Pipelines

Both Standard and Fast have update variants with appended `update_*` workflows.

---

## Entry Points & Orchestration

### CLI Entry Point

**File:** `packages/graphrag/graphrag/cli/index.py`

```python
def index_cli(root_dir, method, verbose, cache, dry_run, skip_validation):
    config = load_config(root_dir=root_dir)
    _run_index(config=config, method=method, is_update_run=False, ...)

def _run_index(config, method, is_update_run, ...):
    validate_config_names(config)  # Tests LLM connectivity
    outputs = asyncio.run(
        api.build_index(config=config, method=method, is_update_run=is_update_run, ...)
    )
```

### API Entry Point

**File:** `packages/graphrag/graphrag/api/index.py`

```python
async def build_index(
    config: GraphRagConfig,
    method: IndexingMethod | str = IndexingMethod.Standard,
    is_update_run: bool = False,
    callbacks: list[WorkflowCallbacks] | None = None,
    additional_context: dict[str, Any] | None = None,
    verbose: bool = False,
    input_documents: pd.DataFrame | None = None,
### Pipeline Runner

**File:** `packages/graphrag/graphrag/index/run/run_pipeline.py`

```python
async def run_pipeline(pipeline, config, callbacks, is_update_run, additional_context, input_documents):
    input_storage = create_storage(config.input_storage)
    output_storage = create_storage(config.output_storage)
    output_table_provider = create_table_provider(config.table_provider, output_storage)
    cache = create_cache(config.cache)

    state_json = await output_storage.get("context.json")
    state = json.loads(state_json) if state_json else {}

    if is_update_run:
        # Create timestamped delta/previous storage hierarchy
        ...

    context = PipelineRunContext(stats, input_storage, output_storage,
                                  output_table_provider, previous_table_provider,
                                  cache, callbacks, state)

    for name, workflow_function in pipeline.run():
        with WorkflowProfiler() as profiler:
            result = await workflow_function(config, context)
        yield PipelineRunResult(workflow=name, result=result.result, ...)
```

### PipelineRunContext

**File:** `packages/graphrag/graphrag/index/typing/context.py`

```python
@dataclass
class PipelineRunContext:
    stats: PipelineRunStats
    input_storage: Storage
    output_storage: Storage
    output_table_provider: TableProvider
    previous_table_provider: TableProvider | None
    cache: Cache
    callbacks: WorkflowCallbacks
    state: PipelineState  # dict[Any, Any]
```

---

## Common Infrastructure

### Storage

**Package:** `graphrag-storage`

| Storage Backend | Type | Best For |
|---|---|---|
| `FileStorage` | Local filesystem | Development / single-machine |
| `BlobStorage` | Azure Blob Storage | Production / cloud |
| `CosmosDBStorage` | Azure Cosmos DB | Production / cloud |

**Table Providers:**
| Provider | Format | Notes |
|---|---|---|
| `ParquetTableProvider` | Parquet files | Default. Efficient columnar format |
| `CSVTableProvider` | CSV files | Text-based, human-readable |

### Cache

**Package:** `graphrag-cache`

| Cache Backend | Type | Best For |
|---|---|---|
| `MemoryCache` | In-memory dict | Development / testing |
| `JSONCache` | JSON file | Simple persistence |
| `SQLiteCache` | SQLite database | Persistent, efficient |
| `NoopCache` | No-op | Disable caching |

### DataReader

**File:** `packages/graphrag/graphrag/data_model/data_reader.py`

```python
reader = DataReader(table_provider)
entities = await reader.entities()
relationships = await reader.relationships()
communities = await reader.communities()
text_units = await reader.text_units()
documents = await reader.documents()
covariates = await reader.covariates()
community_reports = await reader.community_reports()
```

### Row Transformers

**File:** `packages/graphrag/graphrag/data_model/row_transformers.py`

| Transformer | Type Coercions |
|---|---|
| `transform_entity_row` | human_readable_id→int, text_unit_ids→list, frequency→int, degree→int |
| `transform_relationship_row` | human_readable_id→int, weight→float, text_unit_ids→list |
| `transform_text_unit_row` | human_readable_id→int, n_tokens→int, ids→list |
| `transform_document_row` | human_readable_id→int, text_unit_ids→list |
| `transform_entity_row_for_embedding` | Adds `title_description` column |
| `transform_community_row` | human_readable_id→int, community→int, level→int, children→list |
| `transform_community_report_row` | human_readable_id→int, community→int, rank→float, findings→list |
| `transform_covariate_row` | human_readable_id→int |

---

## Step 1: Load Input Documents

**Workflow:** `load_input_documents`
**File:** `packages/graphrag/graphrag/index/workflows/load_input_documents.py`

### Purpose
Read source documents from storage, parse them into a standardized format.

### Process
```python
async def run_workflow(config, context):
    input_reader = create_input_reader(config.input, context.input_storage)
    async with context.output_table_provider.open("documents") as documents_table:
        sample, total_count = await load_input_documents(input_reader, documents_table)
        context.stats.num_documents = total_count
```

### Document Schema (Raw)

| Column | Type | Description |
|---|---|---|
| `id` | str | Unique document identifier |
| `title` | str | Document title |
| `text` | str | Document text content |
| `creation_date` | str (optional) | Document creation date |
| `raw_data` | str (optional) | Raw source data |
| `human_readable_id` | int | Sequential counter |

### Key Operations
1. Create an `InputReader` via factory
2. Iterate documents asynchronously, convert each to dict
3. Assign sequential `human_readable_id`
4. Write each row to `documents` table
5. Raise `ValueError` if 0 documents loaded
) -> list[PipelineRunResult]:
```
## Step 2: Create Base Text Units (Chunking)

**Workflow:** `create_base_text_units`
**File:** `packages/graphrag/graphrag/index/workflows/create_base_text_units.py`

### Purpose
Split documents into smaller chunks (text units) for LLM processing.

### Process
```python
async def run_workflow(config, context):
    tokenizer = get_tokenizer(encoding_model=config.chunking.encoding_model)
    chunker = create_chunker(config.chunking, tokenizer.encode, tokenizer.decode)
    async with (
        context.output_table_provider.open("documents") as documents_table,
        context.output_table_provider.open("text_units") as text_units_table,
    ):
        sample_rows = await create_base_text_units(
            documents_table, text_units_table, total_rows,
            context.callbacks, tokenizer, chunker,
            prepend_metadata=config.chunking.prepend_metadata,
        )
```

### Chunking Strategies

| Strategy | Class | Description |
|---|---|---|
| **Token** | `TokenChunker` | Splits by token count with overlap (Default) |
| **Sentence** | `SentenceChunker` | Splits by sentence boundaries |

### Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `type` | `Tokens` | Chunking strategy |
| `size` | 1200 | Target chunk size in tokens |
| `overlap` | 100 | Overlap between chunks in tokens |
| `encoding_model` | `o200k_base` | Tokenizer encoding model |
| `prepend_metadata` | `None` | Optional metadata to prepend |

### Text Unit Schema (Raw)

| Column | Type | Description |
|---|---|---|
| `id` | str | SHA-512 hash of text content |
| `document_id` | str | Source document ID |
| `text` | str | Chunk text content |
| `n_tokens` | int | Token count |

---

## Step 3: Create Final Documents

**Workflow:** `create_final_documents`
**File:** `packages/graphrag/graphrag/index/workflows/create_final_documents.py`

### Purpose
Enrich documents with `text_unit_ids` — the document→chunk mapping.

### Process
```python
async def run_workflow(_config, context):
    async with (
        context.output_table_provider.open("text_units") as text_units_table,
        context.output_table_provider.open("documents", transformer=transform_document_row) as documents_table,
        context.output_table_provider.open("documents") as output_table,
    ):
        sample = await create_final_documents(text_units_table, documents_table, output_table)
```

### Final Document Schema

| Column | Type | Description |
|---|---|---|
| `id` | str | Document ID |
| `human_readable_id` | int | Sequential ID |
| `title` | str | Document title |
| `text` | str | Full document text |
| `text_unit_ids` | list[str] | IDs of chunks belonging to this document |
| `creation_date` | str | Document creation date |
| `raw_data` | str (optional) | Raw source data |

### Key Operations
1. Build mapping: `document_id → [text_unit_id, ...]` from text units table
2. Read documents, enrich each with its `text_unit_ids`
3. Write enriched documents to output table with type coercion
## Step 4: Extract Entity Graph

**Workflow:** `extract_graph`
**File:** `packages/graphrag/graphrag/index/workflows/extract_graph.py`

### Purpose
Use an LLM to extract entities and relationships from text units.

### Process
```python
async def run_workflow(config, context):
    reader = DataReader(context.output_table_provider)
    text_units = await reader.text_units()

    extraction_model = create_completion(
        config.get_completion_model_config(config.extract_graph.completion_model_id),
        cache=context.cache.child(config.extract_graph.model_instance_name),
        cache_key_creator=cache_key_creator,
    )
    summarization_model = create_completion(
        config.get_completion_model_config(config.summarize_descriptions.completion_model_id),
        ...,
    )

    entities, relationships, raw_entities, raw_relationships = await extract_graph(
        text_units=text_units,
        extraction_model=extraction_model,
        extraction_prompt=extraction_prompts.extraction_prompt,
        entity_types=config.extract_graph.entity_types,
        max_gleanings=config.extract_graph.max_gleanings,
        summarization_model=summarization_model,
        max_summary_length=config.summarize_descriptions.max_length,
        max_input_tokens=config.summarize_descriptions.max_input_tokens,
    )

    await context.output_table_provider.write_dataframe("entities", entities)
    await context.output_table_provider.write_dataframe("relationships", relationships)
    if config.snapshots.raw_graph:
        await context.output_table_provider.write_dataframe("raw_entities", raw_entities)
        await context.output_table_provider.write_dataframe("raw_relationships", raw_relationships)
```

### Two-Phase Extraction

**Phase 1 — Initial Extraction:** LLM identifies entities + relationships per text unit. Multiple "gleaning" passes refine (`max_gleanings`, default: 1).

**Phase 2 — Summarization:** LLM summarizes descriptions, merges back.

### Entity Schema

| Column | Type | Description |
|---|---|---|
| `id` | str | Unique entity ID |
| `title` | str | Entity name |
| `type` | str | Entity type (organization, person, geo, event) |
| `description` | str | LLM-generated description |
| `text_unit_ids` | list[str] | Source chunk IDs |
| `frequency` | int | Occurrence count |
| `human_readable_id` | int | Sequential ID |
| `degree` | int | Node degree (added in Step 6) |

### Relationship Schema

| Column | Type | Description |
|---|---|---|
| `id` | str | Unique ID |
| `source` | str | Source entity title |
| `target` | str | Target entity title |
| `description` | str | LLM-generated description |
| `weight` | float | Relationship weight (default: 1.0) |
| `text_unit_ids` | list[str] | Source chunk IDs |
| `human_readable_id` | int | Sequential ID |
| `combined_degree` | int | Sum of source+target degrees (added in Step 6) |

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `extract_graph.entity_types` | `["organization","person","geo","event"]` | Entity types |
| `extract_graph.max_gleanings` | 1 | Refinement passes |
| `summarize_descriptions.max_length` | 500 | Max summary length |
| `summarize_descriptions.max_input_tokens` | 4000 | Max input tokens |
## Step 4 (NLP Variant): Extract Graph via NLP

**Workflow:** `extract_graph_nlp`
**File:** `packages/graphrag/graphrag/index/workflows/extract_graph_nlp.py`

### Purpose
Alternative to LLM extraction — uses NLP noun phrase extraction. Faster and cheaper.

### Process
```python
async def run_workflow(config, context):
    text_analyzer = create_noun_phrase_extractor(config.extract_graph_nlp.text_analyzer)
    async with (
        context.output_table_provider.open("text_units", truncate=False) as text_units_table,
        context.output_table_provider.open("entities") as entities_table,
        context.output_table_provider.open("relationships") as relationships_table,
    ):
        result = await extract_graph_nlp(
            text_units_table, context.cache,
            entities_table, relationships_table,
            text_analyzer=text_analyzer,
            normalize_edge_weights=config.extract_graph_nlp.normalize_edge_weights,
        )
```

### Noun Phrase Extractors

| Extractor | Type | Description |
|---|---|---|
| `RegexEnglish` | `regex_english` | Fast regex-based (English only) |
| `Syntactic` | `syntactic_parser` | SpaCy dependency parsing + NER |
| `CFG` | `cfg` | CFG-based noun chunk extraction + NER |

### Key Operations
1. Create noun phrase extractor from config
2. Build noun graph from text units
3. Entities = noun phrases (type: "NOUN PHRASE")
4. Relationships = co-occurrence edges between noun phrases

---

## Step 5: Prune Graph (Fast mode only)

**Workflow:** `prune_graph`
**File:** `packages/graphrag/graphrag/index/workflows/prune_graph.py`

### Purpose
Remove low-quality entities/relationships from NLP extraction.

### Pruning Parameters

| Parameter | Default | Description |
|---|---|---|
| `min_node_freq` | 0 | Minimum node frequency |
| `max_node_freq_std` | None | Max std devs from mean frequency |
| `min_node_degree` | 0 | Minimum node degree |
| `max_node_degree_std` | None | Max std devs from mean degree |
| `min_edge_weight_pct` | 0.0 | Minimum edge weight percentile |
| `remove_ego_nodes` | False | Remove ego nodes |
| `lcc_only` | False | Keep only largest connected component |

---

## Step 6: Finalize Graph

**Workflow:** `finalize_graph`
**File:** `packages/graphrag/graphrag/index/workflows/finalize_graph.py`

### Purpose
Compute node degrees and finalize entity/relationship records. Optionally exports GraphML.

### Process
```python
async def run_workflow(config, context):
    async with (
        context.output_table_provider.open("entities", transformer=transform_entity_row) as entities_table,
        context.output_table_provider.open("relationships", transformer=transform_relationship_row) as rels_table,
    ):
        result = await finalize_graph(entities_table, rels_table)

    if config.snapshots.graphml:
        rels = await context.output_table_provider.read_dataframe("relationships")
        await snapshot_graphml(rels, name="graph", storage=context.output_storage)
```

### Algorithm
1. **Build degree map:** Stream relationships, count per entity (undirected, deduplicated)
2. **Finalize entities:** Add `degree` field
3. **Finalize relationships:** Add `combined_degree` (source_degree + target_degree)
4. **Optional:** Export GraphML for visualization (Gephi, etc.)

---
## Step 7: Extract Covariates (Claims)

**Workflow:** `extract_covariates`
**File:** `packages/graphrag/graphrag/index/workflows/extract_covariates.py`

### Purpose
Optionally extract claims/covariates from text units.

### Process
```python
async def run_workflow(config, context):
    if config.extract_claims.enabled:
        reader = DataReader(context.output_table_provider)
        text_units = await reader.text_units()
        model = create_completion(model_config, cache=..., cache_key_creator=...)
        output = await extract_covariates(
            text_units, model=model, covariate_type="claim",
            max_gleanings=config.extract_claims.max_gleanings,
            claim_description=config.extract_claims.description,
            prompt=prompts.extraction_prompt,
        )
        await context.output_table_provider.write_dataframe("covariates", output)
```

### Covariate Schema

| Column | Type | Description |
|---|---|---|
| `id` | str | UUID |
| `human_readable_id` | int | Sequential ID |
| `covariate_type` | str | Type (e.g., "claim") |
| `type` | str | Entity type involved |
| `description` | str | Claim description |
| `subject_id` | str | Subject entity ID |
| `object_id` | str | Object entity ID |
| `status` | str | Claim status |
| `start_date` / `end_date` | str | Date range |
| `source_text` | str | Source text evidence |
| `text_unit_id` | str | Source chunk ID |

### Configuration
```python
config.extract_claims.enabled  # Default: False
config.extract_claims.max_gleanings  # Default: 1
```

---

## Step 8: Create Communities

**Workflow:** `create_communities`
**File:** `packages/graphrag/graphrag/index/workflows/create_communities.py`

### Purpose
Apply hierarchical community detection (Leiden algorithm) to create multi-level communities.

### Process
```python
async def run_workflow(config, context):
    reader = DataReader(context.output_table_provider)
    relationships = await reader.relationships()
    async with (
        context.output_table_provider.open("entities") as entities_table,
        context.output_table_provider.open("communities") as communities_table,
    ):
        sample_rows = await create_communities(
            communities_table, entities_table, relationships,
            max_cluster_size=config.cluster_graph.max_cluster_size,  # Default: 10
            use_lcc=config.cluster_graph.use_lcc,  # Default: True
            seed=config.cluster_graph.seed,  # Default: 0xDEADBEEF
        )
```

### Community Schema

| Column | Type | Description |
|---|---|---|
| `id` | str | UUID |
| `human_readable_id` | int | Community number |
| `community` | int | Cluster ID |
| `level` | int | Hierarchy level (0=top) |
| `parent` | int | Parent community ID |
| `children` | list[int] | Child community IDs |
| `title` | str | "Community {id}" |
| `entity_ids` | list[str] | Entity IDs in this community |
| `relationship_ids` | list[str] | Relationship IDs |
| `text_unit_ids` | list[str] | Chunk IDs referenced |
| `period` | str | ISO date of creation |
| `size` | int | Number of entities |

### Algorithm
- **Library:** `graspologic-native` (pinned to 1.2.x)
- **Algorithm:** Hierarchical Leiden clustering
- **Input:** Weighted relationship graph
## Step 9: Create Final Text Units

**Workflow:** `create_final_text_units`
**File:** `packages/graphrag/graphrag/index/workflows/create_final_text_units.py`

### Purpose
Enrich text units with entity, relationship, and covariate ID references.

### Process
```python
async def run_workflow(config, context):
    async with (
        context.output_table_provider.open("text_units", transformer=...) as text_units_table,
        context.output_table_provider.open("entities", transformer=...) as entities_table,
        context.output_table_provider.open("relationships", transformer=...) as relationships_table,
        context.output_table_provider.open("text_units") as output_table,
        cov_ctx as covariates_table,
    ):
        sample = await create_final_text_units(
            text_units_table, entities_table, relationships_table,
            output_table, covariates_table,
        )
```

### Key Operations
1. Build reverse mappings: `text_unit_id → [entity_id, ...]`, etc.
2. Stream text units, enrich with entity/relationship/covariate IDs
3. Assign sequential `human_readable_id`
4. Write enriched rows

### Final Text Unit Schema

| Column | Type | Description |
|---|---|---|
| `id` | str | Chunk ID (SHA-512 hash) |
| `human_readable_id` | int | Sequential ID |
| `text` | str | Chunk text |
| `n_tokens` | int | Token count |
| `document_id` | str | Source document ID |
| `entity_ids` | list[str] | Entities found in this chunk |
| `relationship_ids` | list[str] | Relationships in this chunk |
| `covariate_ids` | list[str] | Covariates in this chunk |

---

## Step 10: Create Community Reports

Two variants exist:

### Variant A: Graph-Based (Standard mode)

**Workflow:** `create_community_reports`
**File:** `packages/graphrag/graphrag/index/workflows/create_community_reports.py`

Uses graph context (entities + relationships) for report generation:
```python
async def run_workflow(config, context):
    reader = DataReader(context.output_table_provider)
    relationships = await reader.relationships()
    entities = await reader.entities()
    communities = await reader.communities()
    model = create_completion(model_config, cache=..., cache_key_creator=...)

    output = await create_community_reports(
        relationships, entities, communities, claims,
        model=model, prompt=prompts.graph_prompt,
        max_input_length=config.community_reports.max_input_length,
        max_report_length=config.community_reports.max_length,
    )
    await context.output_table_provider.write_dataframe("community_reports", output)
```

### Variant B: Text-Based (Fast mode)

**Workflow:** `create_community_reports_text`
**File:** `packages/graphrag/graphrag/index/workflows/create_community_reports_text.py`

Uses text unit context instead of graph context:
```python
output = await create_community_reports_text(
    entities, communities, text_units,
    model=model, prompt=prompts.text_prompt, ...
)
```

### Community Report Schema

| Column | Type | Description |
|---|---|---|
| `id` | str | UUID |
| `human_readable_id` | int | Sequential ID |
| `community` | int | Community ID |
| `level` | int | Hierarchy level |
| `parent` | int | Parent community ID |
| `children` | list[int] | Child community IDs |
| `title` | str | Report title |
| `summary` | str | Executive summary |
| `full_content` | str | Full report content |
| `rank` | float | Importance rating |
| `rating_explanation` | str | Explanation of rating |
| `findings` | list | Structured findings |
| `full_content_json` | str | JSON-encoded report |
| `period` | str | ISO date |
| `size` | int | Number of entities |

### Configuration
```python
config.community_reports.max_length        # Default: 2000 tokens
config.community_reports.max_input_length  # Default: 8000 tokens
```
## Step 11: Generate Text Embeddings

**Workflow:** `generate_text_embeddings`
**File:** `packages/graphrag/graphrag/index/workflows/generate_text_embeddings.py`

### Purpose
Generate vector embeddings for semantic search across text units, entities, and community reports.

### Embedding Fields

| Field Name | Source Table | Embed Column | Purpose |
|---|---|---|---|
| `text_unit_text` | `text_units` | `text` | Semantic search over chunks |
| `entity_description` | `entities` | `title_description` | Semantic search over entities |
| `community_full_content` | `community_reports` | `full_content` | Semantic search over communities |

### Process
```python
async def run_workflow(config, context):
    model_config = config.get_embedding_model_config(config.embed_text.embedding_model_id)
    model = create_embedding(model_config, cache=..., cache_key_creator=...)
    tokenizer = model.tokenizer

    await generate_text_embeddings(
        config=config, table_provider=context.output_table_provider,
        callbacks=context.callbacks, model=model, tokenizer=tokenizer,
    )
```

### Vector Store Types

| Type | Backend |
|---|---|
| `AzureAISearch` | Azure AI Search |
| `CosmosDB` | Azure Cosmos DB for MongoDB vCore |
| `LanceDB` | LanceDB (local) — Default |

### Configuration
```python
config.embed_text.names              # Which embeddings (default: all 3)
config.embed_text.batch_size         # Default: 16
config.embed_text.batch_max_tokens   # Default: 8192
config.vector_store.type             # Vector store backend
config.vector_store.vector_size      # Embedding dimension
config.snapshots.embeddings          # Save to Parquet (default: False)
```

---

## Incremental Update Pipeline

### Overview
The update pipeline processes only new/changed documents and merges results with the existing index. Triggered by `is_update_run=True`.

### Update Detection
Documents are compared by **title** — new titles are processed; existing titles are ignored.

```python
async def get_delta_docs(input_dataset, table_provider):
    final_docs = await table_provider.read_dataframe("documents")
    previous_docs = final_docs["title"].unique().tolist()
    dataset_docs = input_dataset["title"].unique().tolist()
    new_docs = input_dataset.loc[~input_dataset["title"].isin(previous_docs)]
    return InputDelta(new_docs, ...)
```

### Update Storage Hierarchy
```
update_output/
  └── {timestamp}/
      ├── delta/        # New index output
      ├── previous/     # Copy of previous full index
      └── final/        # Merged output
```
### Update Workflow Sequence

After standard workflows run on delta documents, these merge workflows execute:

| Workflow | Purpose | Files |
|---|---|---|
| `update_final_documents` | Concatenate old + new documents | `workflows/update_final_documents.py` |
| `update_entities_relationships` | Merge entities (de-dup titles), merge relationships, re-summarize | `workflows/update_entities_relationships.py` |
| `update_text_units` | Merge text units with entity ID remapping | `workflows/update_text_units.py` |
| `update_covariates` | Merge covariates | `workflows/update_covariates.py` |
| `update_communities` | Merge communities with ID remapping | `workflows/update_communities.py` |
| `update_community_reports` | Merge community reports | `workflows/update_community_reports.py` |
| `update_text_embeddings` | Regenerate embeddings for merged data | `workflows/update_text_embeddings.py` |
| `update_clean_state` | Clean up temporary state variables | `workflows/update_clean_state.py` |

---

## Output Data Schema

### Final Output Tables (written as Parquet files)

| Table | File Name | Contents |
|---|---|---|
| `documents` | `output/documents.parquet` | Documents with chunk mappings |
| `text_units` | `output/text_units.parquet` | Text chunks with entity/relationship/covariate mappings |
| `entities` | `output/entities.parquet` | Entity nodes with descriptions and degrees |
| `relationships` | `output/relationships.parquet` | Relationship edges with weights and descriptions |
| `communities` | `output/communities.parquet` | Community hierarchy |
| `community_reports` | `output/community_reports.parquet` | LLM-generated community summaries |
| `covariates` | `output/covariates.parquet` | Extracted claims (if enabled) |

### Optional Snapshots

| Snapshot | Config Flag | File | Description |
|---|---|---|---|
| Raw Entities | `snapshots.raw_graph` | `raw_entities.parquet` | Pre-summarization entities |
| Raw Relationships | `snapshots.raw_graph` | `raw_relationships.parquet` | Pre-summarization relationships |
| GraphML | `snapshots.graphml` | `graph.graphml` | Graph for visualization |
| Embeddings | `snapshots.embeddings` | `embeddings.{name}.parquet` | Vector embeddings |

### Runtime Files

| File | Description |
|---|---|
| `output/stats.json` | Pipeline run statistics (timing, memory) |
| `output/context.json` | Pipeline state (for resumability) |

---

## Configuration Reference

### Minimal settings.yaml

```yaml
input:
  file_type: text
  base_dir: input

llm:
  model: gpt-4.1
  api_key: ${GRAPHRAG_API_KEY}

embeddings:
  llm:
    model: text-embedding-3-large
    api_key: ${GRAPHRAG_API_KEY}

chunking:
  size: 1200
  overlap: 100

extract_graph:
  entity_types: ["organization", "person", "geo", "event"]
  max_gleanings: 1

summarize_descriptions:
  max_length: 500

community_reports:
  max_length: 2000
  max_input_length: 8000

cluster_graph:
  max_cluster_size: 10
  use_lcc: true
  seed: 3735928559

embed_text:
  names:
    - text_unit_text
    - entity_description
    - community_full_content

vector_store:
  type: lancedb
  db_uri: output/lancedb

snapshots:
  embeddings: false
  graphml: false
  raw_graph: false
```
### Key Config Models

| Config Model | File | Purpose |
|---|---|---|
| `GraphRagConfig` | `config/models/graph_rag_config.py` | Root config with all sub-configs |
| `InputConfig` | `graphrag-input` | Input document source |
| `StorageConfig` | `graphrag-storage` | Storage backend settings |
| `ChunkingConfig` | `graphrag-chunking` | Text chunking parameters |
| `ExtractGraphConfig` | `config/models/extract_graph_config.py` | LLM entity extraction |
| `ExtractGraphNLPConfig` | `config/models/extract_graph_nlp_config.py` | NLP extraction settings |
| `SummarizeDescriptionsConfig` | `config/models/summarize_descriptions_config.py` | Description summarization |
| `CommunityReportsConfig` | `config/models/community_reports_config.py` | Community report generation |
| `ClusterGraphConfig` | `config/models/cluster_graph_config.py` | Community detection |
| `EmbedTextConfig` | `config/models/embed_text_config.py` | Text embedding |
| `PruneGraphConfig` | `config/models/prune_graph_config.py` | Graph pruning (Fast mode) |
| `ExtractClaimsConfig` | `config/models/extract_claims_config.py` | Claim extraction |
| `VectorStoreConfig` | `graphrag-vectors` | Vector store settings |
| `CacheConfig` | `graphrag-cache` | Cache settings |
| `SnapshotsConfig` | `config/models/snapshots_config.py` | Optional snapshot outputs |

---

## End-to-End Data Flow Diagram

```
                       ┌─────────────────────┐
                       │   Input Documents    │
                       │   (files, blob...)   │
                       └──────────┬──────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  load_input_documents   │
                    │  Read & parse docs      │
                    │  Write to "documents"   │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ create_base_text_units  │
                    │  Chunk docs by tokens   │
                    │  Write to "text_units"  │
                    └──────────┬──────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
          ┌──────────────────┐  ┌──────────────────────┐
          │  extract_graph   │  │  extract_graph_nlp   │
          │  (Standard)      │  │  (Fast)              │
          │  LLM: entities + │  │  NLP: noun phrases + │
          │  relationships   │  │  co-occurrence edges │
          └────────┬─────────┘  └──────────┬───────────┘
                   │                       │
                   │                       ▼
                   │              ┌──────────────────┐
                   │              │   prune_graph    │
                   │              │  (Fast only)     │
                   │              │  Remove noise    │
                   │              └────────┬─────────┘
                   │                       │
                   └──────────┬────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │     finalize_graph      │
                    │  Compute degrees        │
                    │  Export GraphML (opt)   │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │  extract_covariates     │
                    │  LLM: claim extraction  │
                    │  (if enabled)           │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │   create_communities    │
                    │  Leiden clustering      │
                    │  Build hierarchy        │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ create_final_text_units │
                    │  Enrich chunks with     │
                    │  entity/rel/cov IDs     │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ create_community_reports│
                    │  LLM: generate reports  │
                    │  for each community     │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │generate_text_embeddings │
                    │  Embed text units,      │
                    │  entities, community    │
                    │  reports → vector store │
                    └─────────────────────────┘
```