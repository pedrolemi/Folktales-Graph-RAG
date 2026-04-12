from loguru import logger
from models.folktale import Folktale

def filter_valid_folktales(folktales: list[Folktale], min_events: int):
    valid_folktales = []

    for folktale in folktales:
        n_events = len(folktale.events)

        if n_events <= min_events:
            logger.warning(
                f"'{folktale.title}' has too few events "
                f"({n_events}, minimum required: {min_events})"
            )
        else:
            valid_folktales.append(folktale)

    logger.info(f"Valid folktales: {len(valid_folktales)}/{len(folktales)}")

    return valid_folktales