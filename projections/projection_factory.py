from projections.base_projection_layer import BaseProjectionLayer
from projections.kl_projection_layer import KLProjectionLayer
from projections.w2_projection_layer import WassersteinProjectionLayer


def get_projection_layer(proj_type: str = "", **kwargs) -> BaseProjectionLayer:
    """
    Factory to generate the projection layers for all projections.
    Args:
        proj_type: One of 'w2' and 'kl'
        **kwargs: arguments for projection layer

    Returns:

    """
    if proj_type.lower() == "w2":
        return WassersteinProjectionLayer(proj_type, **kwargs)

    elif proj_type.lower() == "kl":
        return KLProjectionLayer(proj_type, **kwargs)

    else:
        raise ValueError(
            f"Invalid projection type {proj_type}."
            f" Choose one of None/' ', 'ppo', 'papi', 'w2', 'w2_non_com', 'frob', 'kl', or 'entropy'.")
