from pptagent.research import DeepResearchAdapter
adapter = DeepResearchAdapter()
result = adapter.resolve_source_pdf(
    url="https://ai.meta.com/research/publications/flow-matching-guide-and-code/",
    download_dir=r"/data/group/zhaolab/project/PresentnAgent/PresentAgent/research_pdfs",
    topic="flow matching",
)

print(result.to_dict())
