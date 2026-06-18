from pathlib import Path
from ..data.ZeusDatasetSample import ZeusDatasetSample


class TokenMap:
    """Maps model output indexes to LMX tokens and vice versa."""
    
    def __init__(
            self,
            tokens: list[str],
            bos_token_index: int,
            eos_token_index: int,
            unknown_token_index: int,
    ):
        """Use static methods to load or create token maps,
        instead of using the constructor directly."""

        # check invariants about the input
        assert len(set(tokens)) == len(tokens), \
            "Token vocabulary may not contain duplicates"
        assert all(token == token.strip() for token in tokens), \
            "Tokens must not contain whitespace at the start and end"
        assert bos_token_index >= 0 and bos_token_index < len(tokens)
        assert eos_token_index >= 0 and eos_token_index < len(tokens)
        assert unknown_token_index >= 0 and unknown_token_index < len(tokens)

        self.tokens = tokens
        """Ordered list of LMX tokens. Position of the token
        in the list defines its index in the zeus output layer."""

        self.inverted_lookup_map: dict[str, int] = {
            token: index
            for index, token in enumerate(self.tokens)
        }
        """Dictionary that can be used to get index for a token fast."""

        self.bos_token = tokens[bos_token_index]
        """BOS (beginning of sequence) token, used as the first input token
        into the autoregressive decoder."""

        self.bos_token_index = bos_token_index
        """Index of the BOS (beginning of sequence) token"""

        self.eos_token = tokens[eos_token_index]
        """EOS (end of sequence) token, produced by the autoregressive
        decoder when the generation of output is complete."""

        self.eos_token_index = eos_token_index
        """Index of the EOS (end of sequence) token"""

        self.unknown_token = tokens[unknown_token_index]
        """Unknown token, used when training/finetuning data contains
        unknown token that cannot be encoded (out of vocabulary token)."""

        self.unknown_token_index = unknown_token_index
        """Index of the unknown (out of vocabulary token) token"""
    
    def token_to_index(self, token: str, allow_unknown_tokens=False) -> int:
        """Converts a LMX token to model feature index"""
        index = self.inverted_lookup_map.get(token)

        if index is None:
            if allow_unknown_tokens:
                return self.unknown_token_index
            else:
                raise Exception(
                    f"Token '{token}' is not known to the model."
                )

        return index

    def index_to_token(self, index: int) -> str:
        """Decodes a model index into a LMX token"""
        if index >= 0 and index < len(self.tokens):
            return self.tokens[index]
        else:
            raise Exception(
                f"Given index {index} is out of range of known tokens."
            )
    
    def indices_to_lmx(self, indices: list[int]) -> str:
        """Decodes a 1D sequence of token indices to an LMX string"""
        tokens = [
            self.index_to_token(index)
            for index in indices
            if index not in [
                self.bos_token_index,
                self.eos_token_index
            ]
        ]
        return " ".join(tokens)
    
    def __len__(self) -> int:
        """Returns the number of tokens in the map, which corresponds
        exactly to the size of the model's output layer."""
        return len(self.tokens)
    
    @staticmethod
    def create_from_dataset(samples: list[ZeusDatasetSample]) -> "TokenMap":
        """Creates a map for only those tokens that are present in the
        given (training) dataset. Useful when training a new model."""
        tokens = ["<bos/eos>", "<unk>"]
        
        for sample in samples:
            for token in sample.lmx.split():
                if token not in tokens:
                    tokens.append(token)

        return TokenMap(
            tokens=tokens,
            bos_token_index=0,
            eos_token_index=0,
            unknown_token_index=1
        )
    
    @staticmethod
    def create_for_full_lmx_vocabulary() -> "TokenMap":
        """Creates a map for all the tokens in the LMX vocabulary,
        as defined by the LMX package. May change version to version
        so still needs to be persisted."""
        import lmx.tokenization.vocabulary
        return TokenMap(
            tokens=["<bos/eos>", "<unk>"] +
                lmx.tokenization.vocabulary.ALL_TOKENS,
            bos_token_index=0,
            eos_token_index=0,
            unknown_token_index=1
        )

    @staticmethod
    def load_from_file(file_path: Path, legacy: bool) -> "TokenMap":
        """Loads token map from a txt file with one token per line
        ordered in the order the tokens are indexed."""
        tokens = file_path.read_text().splitlines()
        tokens = [token.strip() for token in tokens]

        # the old 2024 models had <bos/eos> internally within the
        # keras model and the model did shift by one index for all
        # inputs and outputs. If we use those models now, we need to have
        # this token and the shift be explicit in the map.
        if legacy:
            assert tokens[0] == "<unk>"
            return TokenMap(
                tokens=["<bos/eos>"] + tokens,
                bos_token_index=0,
                eos_token_index=0,
                unknown_token_index=1
            )

        # The new 2026 models have all the tokens explicitly in the map
        else:
            assert tokens[0] == "<bos/eos>"
            assert tokens[1] == "<unk>"
            return TokenMap(
                tokens=tokens,
                bos_token_index=0,
                eos_token_index=0,
                unknown_token_index=1
            )
    
    @staticmethod
    def load_from_model_folder(model_folder_path: Path) -> "TokenMap":
        """Loads token map from a stored model, given its folder path"""
        # legacy file
        legacy_path = model_folder_path / "tags.txt"
        if legacy_path.exists():
            return TokenMap.load_from_file(legacy_path, legacy=True)
        
        # current file
        return TokenMap.load_from_file(
            model_folder_path / "token_map.txt",
            legacy=False
        )
    
    def write_to_model_folder(self, model_folder_path: Path):
        """Writes the token map into a model folder"""
        self.write_to_file(model_folder_path / "token_map.txt")

    def write_to_file(self, file_path: Path):
        """Persists this map in a txt file"""
        with open(file_path, "w") as file:
            for token in self.tokens:
                file.write(token + "\n")
