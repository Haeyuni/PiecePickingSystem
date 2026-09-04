class PickVerifier:
    def verify(self):
        # No existing RG2 state topic or verified post-lift camera protocol exists in this repository.
        return False, False, 'DRY_RUN_ONLY:RG2_AND_REOBSERVATION_UNAVAILABLE'
