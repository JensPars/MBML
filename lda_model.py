import torch
import pyro
import pyro.distributions as dist

def model(data=None, mask=None, batch_size=None, num_topics=20, num_words=None, num_docs=None, num_words_per_doc=60, device=None):
    """LDA model with masking for variable-length documents."""
    if device is None:
        device = torch.device("cpu")

    if num_words is None:
        raise ValueError("num_words must be specified")
    if num_docs is None:
        raise ValueError("num_docs must be specified")

    with pyro.plate("topics", num_topics):
        topic_words = pyro.sample(
            "topic_words", dist.Dirichlet(torch.ones(num_words, device=device) / num_words)
        )

    with pyro.plate("documents", num_docs, batch_size) as ind:
        data_batch = data[:, ind]
        mask_batch = None
        if mask is not None:
            mask_batch = mask[:, ind]

        doc_topics = pyro.sample(
            "doc_topics", dist.Dirichlet(torch.ones(num_topics, device=device) / num_topics)
        )

        with pyro.plate("words", num_words_per_doc):
            with pyro.poutine.mask(mask=mask_batch):
                word_topics = pyro.sample(
                    "word_topics", dist.Categorical(doc_topics),
                    infer={"enumerate": "parallel"}
                )

                doc_words = pyro.sample(
                    "doc_words",
                    dist.Categorical(topic_words[word_topics]),
                    obs=data_batch
                )

    return topic_words, doc_words

def create_guide(model_arg, num_topics, num_words, num_docs, device=None): # model_arg is unused
    """Creates a custom guide for the LDA model."""
    final_device = device if device is not None else torch.device("cpu")

    def my_local_guide(data=None, mask=None, batch_size=None):
        topic_words_posterior = pyro.param(
                "topic_words_posterior",
                lambda: torch.ones(num_topics, num_words, device=final_device),
                constraint=dist.constraints.positive)

        with pyro.plate("topics", num_topics):
            pyro.sample("topic_words", dist.Dirichlet(topic_words_posterior))

        doc_topics_posterior = pyro.param(
                "doc_topics_posterior",
                lambda: torch.ones(num_docs, num_topics, device=final_device),
                constraint=dist.constraints.simplex)

        with pyro.plate("documents", num_docs, batch_size) as ind:
            pyro.sample("doc_topics", dist.Delta(doc_topics_posterior[ind], event_dim=1))
    return my_local_guide
