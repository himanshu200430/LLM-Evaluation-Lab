def estimate_cost(tokens_in, tokens_out, price_in, price_out):
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000