
import torch
import pyro
import pyro.distributions as dist


def model(data=None, mask=None, num_topics=None, num_words=None, batch_size=None, ratings=None, num_classes=None, device="cpu"):
    """LDA model with masking for variable-length documents."""
    with pyro.plate("topics", num_topics):
        topic_words = pyro.sample(
            "topic_words", dist.Dirichlet(0.1*torch.ones(num_words) / num_words)
        )
    
    # Ordinal regression parameters (priors)
    regression_coefs = None
    cutpoints = None

    if ratings is not None:
        if num_classes is None:
            raise ValueError("num_classes must be specified if ratings are provided.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1 for ordinal regression.")

        # Priors for regression coefficients (one per topic)
        regression_coefs = pyro.sample(
            "regression_coefs",
            dist.Normal(torch.zeros(num_topics, device=device),
                        torch.ones(num_topics, device=device)).to_event(1)
        )

        # Priors for cutpoints (num_classes - 1 cutpoints, ordered)
        # c_0 ~ Normal
        # c_i = c_{i-1} + exp(delta_i), where delta_i ~ Normal for i > 0
        # This creates c_0 < c_1 < ... < c_{K-2}
        
        # Sample the first cutpoint
        c_0 = pyro.sample("c_0", 
                          dist.Normal(torch.tensor(0., device=device), 
                                      torch.tensor(5., device=device))) 
        
        if num_classes > 2: # If more than one cutpoint is needed (i.e., num_classes - 1 > 1)
            # Sample log differences for the remaining num_classes - 2 cutpoints
            # These correspond to log(c_1 - c_0), log(c_2 - c_1), ...
            log_diffs = pyro.sample(
                "log_diffs",
                dist.Normal(torch.zeros(num_classes - 2, device=device),
                            torch.ones(num_classes - 2, device=device)).to_event(1)
            )
            # Calculate cutpoints: c_0, c_0 + exp(log_diffs_0), c_0 + exp(log_diffs_0) + exp(log_diffs_1), ...
            # These are then c_0, c_1, c_2, ...
            incremental_cutpoints = c_0 + torch.cumsum(torch.exp(log_diffs), dim=-1)
            cutpoints = torch.cat([c_0.unsqueeze(0), incremental_cutpoints])
        elif num_classes == 2: # Exactly one cutpoint: c_0
            cutpoints = c_0.unsqueeze(0)
        # If num_classes == 1, an error was raised earlier. No cutpoints needed.


    with pyro.plate("documents", num_docs, batch_size) as ind:
        data = data[:, ind]  # data: [max_words, batch_size]
        if mask is not None:
            mask = mask[:, ind]  # same shape as data

        ratings_batch = None
        if ratings is not None: # ratings should be (num_docs,)
            ratings_batch = ratings[ind]
        
        doc_topics = pyro.sample(
            "doc_topics", dist.Dirichlet(0.1*torch.ones(num_topics) / num_topics)
        )
        
        # Ordinal regression part
        if ratings is not None and regression_coefs is not None and cutpoints is not None:
            # regression_coefs shape: (num_topics)
            # doc_topics shape: (batch_size, num_topics)
            # eta = (doc_topics * regression_coefs.unsqueeze(0)).sum(dim=-1)
            # Using matmul for clarity:
            eta = torch.matmul(doc_topics, regression_coefs) # (batch_size, num_topics) @ (num_topics) -> (batch_size)
            
            pyro.sample(
                "observed_ratings",
                dist.OrderedLogistic(predictor=eta, cutpoints=cutpoints, validate_args=True),
                obs=ratings_batch # ratings_batch should be (batch_size,) and contain integer classes 0..K-1
            )

        with pyro.plate("words", num_words_per_doc):
            with pyro.poutine.mask(mask=mask):
                word_topics = pyro.sample(
                    "word_topics", dist.Categorical(doc_topics),
                    infer={"enumerate": "parallel"}
                )

                doc_words = pyro.sample(
                    "doc_words",
                    dist.Categorical(topic_words[word_topics]),
                    obs=data
                )


    if ratings is not None:
        return topic_words, doc_words, regression_coefs, cutpoints
    else:
        return topic_words, doc_words
    
    
def my_local_guide(data=None, mask=None, batch_size=None, ratings=None, num_classes=None, device="cpu"): # Added ratings, num_classes, device
    # Guide for topic_words
    topic_words_posterior = pyro.param(
            "topic_words_posterior", # Kept original name
            lambda: torch.ones(num_topics, num_words, device=device), # Added device
            constraint=constraints.positive)

    with pyro.plate("topics", num_topics):
        pyro.sample("topic_words", dist.Dirichlet(topic_words_posterior))

    # Guide for ordinal regression parameters (if model includes them)
    if ratings is not None:
        if num_classes is None:
            raise ValueError("num_classes must be specified for guide if ratings are provided.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1 for ordinal regression in guide.")

        # Guide for regression_coefs
        q_regr_coefs_loc = pyro.param(
            "q_regr_coefs_loc",
            lambda: torch.zeros(num_topics, device=device))
        q_regr_coefs_scale = pyro.param(
            "q_regr_coefs_scale",
            lambda: torch.ones(num_topics, device=device),
            constraint=constraints.positive)
        pyro.sample("regression_coefs",
                    dist.Normal(q_regr_coefs_loc, q_regr_coefs_scale).to_event(1))

        # Guide for c_0
        q_c0_loc = pyro.param(
            "q_c0_loc",
            lambda: torch.tensor(0.0, device=device))
        q_c0_scale = pyro.param(
            "q_c0_scale",
            lambda: torch.tensor(1.0, device=device),
            constraint=constraints.positive)
        pyro.sample("c_0", dist.Normal(q_c0_loc, q_c0_scale))

        if num_classes > 2:
            # Guide for log_diffs
            q_log_diffs_loc = pyro.param(
                "q_log_diffs_loc",
                lambda: torch.zeros(num_classes - 2, device=device))
            q_log_diffs_scale = pyro.param(
                "q_log_diffs_scale",
                lambda: torch.ones(num_classes - 2, device=device),
                constraint=constraints.positive)
            pyro.sample("log_diffs",
                        dist.Normal(q_log_diffs_loc, q_log_diffs_scale).to_event(1))

    # Guide for doc_topics
    doc_topics_posterior = pyro.param(
            "doc_topics_posterior", # Kept original name
            lambda: torch.ones(num_docs, num_topics, device=device), # Added device
            constraint=constraints.positive) # Changed to positive for Dirichlet concentration

    with pyro.plate("documents", num_docs, batch_size) as ind:
        # Sample from a Dirichlet distribution for doc_topics
        pyro.sample("doc_topics", dist.Dirichlet(doc_topics_posterior[ind]))

