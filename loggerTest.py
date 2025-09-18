import logging

logging.basicConfig(
    filename=None,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("transcribe_VAD")

logger.debug("Test")
logger.info("Hallo Welt")
logger.warning("Achtung")
logger.error("Da ist ein Fehler passiert")
logger.critical("Schwerer Fehler!")