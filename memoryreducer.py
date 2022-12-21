# -*- coding: utf-8 -*-
"""
src.DemandForecasting.preparation.memoryreducer
===============================================

This module provides the functionality to reduce memory usage in the pipeline.
"""

import numpy as np
import pandas as pd
import logging

from src.utils.config import Config
import src.utils.color_logger as color_logger
import src.DemandForecasting.pipeline.abstractpipelineobject as apo

log = logging.getLogger(__name__)
log.addHandler(color_logger.ColorHandler())


class MemoryReducer(apo.PipelineObject):
    """
    This class adds the functionality to reduce memory.

    """

    def __init__(self, config: Config):
        apo.PipelineObject.__init__(self, config)

    def __transform__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of the transformation function that covers the transformation functionality of
        memory reducing.

        :param data: The dataframe the will be transformed by this Pipeline Object
        :return: A dataframe that has been transformed by this pipeline object
        """

        data = self.reduce_memory(data)

        log.debug('Completed memory reducing.')

        return data

    @staticmethod
    def reduce_memory(data: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce memory data based on dtypes.

        :param data: master sales data
        :return: filtered sales data
        """

        start_mem_usg = data.memory_usage().sum() / 1024 ** 2
        print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
        na_list = []
        data = data.replace([np.inf, -np.inf], np.nan)

        for col in data.select_dtypes(include=np.number).columns:
            # make variables for Int, max and min
            is_int = False
            mx = data[col].max()
            mn = data[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(data[col]).all():
                na_list.append(col)
                data[col].fillna(mn - 1, inplace=True)

            # test if column can be converted to an integer
            print(col)
            as_int = data[col].fillna(-99).astype(np.int64)
            result = (data[col] - as_int)
            result = result.sum()
            if (result > -0.01) and (result < 0.01):
                is_int = True

            # Make Integer/unsigned Integer datatypes
            if is_int:
                if mn >= 0:
                    if mx < 255:
                        data[col] = data[col].astype(np.uint8)
                    elif mx < 65535:
                        data[col] = data[col].astype(np.uint16)
                    elif mx < 4294967295:
                        data[col] = data[col].astype(np.uint32)
                    else:
                        data[col] = data[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        data[col] = data[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        data[col] = data[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        data[col] = data[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        data[col] = data[col].astype(np.int64)

            # Make float data types 32 bit
            else:
                data[col] = data[col].astype(np.float32)

        # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = data.memory_usage().sum() / 1024 ** 2
        print("Memory usage is: ", mem_usg, " MB")
        print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
        return data
