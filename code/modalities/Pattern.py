import tensorflow as tf
from typing import Union, Tuple, List, Type, Dict, Optional, Callable

from modalities import Modality, ModalityLoadInfo


class Pattern(object):
    def __init__(self,
                 *elements: Union[Tuple, ModalityLoadInfo, str],
                 output_map: Callable = None):
        self.elements = elements
        self.output_map = output_map
        self._output_count = len(elements) if not output_map else None
        self._modality_types: Optional[Tuple[Type[Modality]]] = None
        self._modality_ids: Optional[Tuple[str]] = None
        self._flattened: Optional[Tuple[ModalityLoadInfo]] = None
        self._as_dict: Optional[Dict[Type[Modality], List[ModalityLoadInfo]]] = None
        self._contains_labels: Optional[bool] = None

    # region Properties
    @property
    def required_modality_types(self) -> Tuple[Type[Modality]]:
        return self.modality_types

    @property
    def modality_types(self) -> Tuple[Type[Modality]]:
        if self._modality_types is None:
            self._modality_types = tuple(self._extract_modalities_types(self.elements))
        return self._modality_types

    @property
    def modality_ids(self) -> Tuple[str]:
        if self._modality_ids is None:
            ids: List[str] = [modality_type.id() for modality_type in self.modality_types]
            self._modality_ids = tuple(ids)
        return self._modality_ids

    @property
    def flattened(self) -> Tuple[Union[ModalityLoadInfo, str], ...]:
        if self._flattened is None:
            self._flattened = tuple(self._flatten_pattern(self.elements))
        return self._flattened

    @property
    def as_dict(self) -> Dict[Type[Modality], List[ModalityLoadInfo]]:
        if self._as_dict is None:
            self._as_dict = {}

            for modality_load_info in self.flattened:
                if isinstance(modality_load_info, str):
                    continue

                modality_type = modality_load_info.modality
                if modality_type in self._as_dict:
                    self._as_dict[modality_type].append(modality_load_info)
                else:
                    self._as_dict[modality_type] = [modality_load_info]

        return self._as_dict

    @property
    def contains_labels(self):
        if self._contains_labels is None:
            self._contains_labels = "labels" in self.flattened
        return self._contains_labels

    @property
    def modalities_per_sample(self):
        count = len(self.modality_types)
        if self.contains_labels:
            count += 1
        return count

    @property
    def output_count(self) -> int:
        if not self._output_count:
            raise AttributeError("You must use this pattern at least once to determine the number of outputs.")
        return self._output_count

    # endregion

    def with_labels(self) -> Optional["Pattern"]:
        return self.__class__(*self.elements, "labels", output_map=self.output_map)

    def with_added_depth(self) -> Optional["Pattern"]:
        return self.__class__(self.elements, output_map=self.output_map)

    def apply(self, modalities: Dict[str, tf.Tensor]):
        modalities = self._apply_pattern(modalities, self.elements)

        if self.output_map is not None:
            modalities = self.output_map(*modalities)

        if not isinstance(modalities, tf.Tensor):
            if len(modalities) == 1:
                modalities = modalities[0]

        if self._output_count is None:
            if isinstance(modalities, tf.Tensor):
                self._output_count = 1
            else:
                self._output_count = len(modalities)

        return modalities

    @staticmethod
    def _apply_pattern(modalities: Dict[str, tf.Tensor],
                       pattern: Union[ModalityLoadInfo, str, Tuple, List]):
        if isinstance(pattern, ModalityLoadInfo):
            modality_id = pattern.modality.id()
            modality = modalities[modality_id]
            modality = modality[:pattern.length]
            # if pattern.output_map is not None:
            #     modality = pattern.output_map(modality, modalities)
            # modality = tf.reshape(modality, pattern.output_shape, name="reshape_to_modality_output_shape")
            return modality
        elif isinstance(pattern, str) and pattern == "labels":
            return modalities["labels"]
        elif isinstance(pattern, tuple):
            return tuple([Pattern._apply_pattern(modalities, x) for x in pattern])
        elif isinstance(pattern, list):
            return [Pattern._apply_pattern(modalities, x) for x in pattern]
        else:
            raise TypeError("Not supported type : {}".format(type(pattern)))

    def __len__(self):
        return len(self.elements)

    def __str__(self):
        return str(self.elements)

    def __iter__(self):
        return iter(self.elements)

    @staticmethod
    def _extract_modalities_types(elements) -> List[Type[Modality]]:
        types: List[Type[Modality]] = []
        for element in elements:
            if isinstance(element, ModalityLoadInfo):
                if element.modality not in types:
                    types.append(element.modality)
            elif not isinstance(element, str):
                element_types = Pattern._extract_modalities_types(element)
                for element_type in element_types:
                    if element_type not in types:
                        types.append(element_type)
        return types

    @staticmethod
    def _flatten_pattern(elements) -> List[Union[ModalityLoadInfo, str]]:
        if isinstance(elements, ModalityLoadInfo) or isinstance(elements, str):
            return [elements]

        return sum([[x] if (isinstance(x, str) or isinstance(x, ModalityLoadInfo))
                    else Pattern._flatten_pattern(x)
                    for x in elements],
                   [])
