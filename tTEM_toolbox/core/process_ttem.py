# process_ttem.py
# Version 11.18.2023
# Created: 2023-11-17
# Author: Jiawei Li
import pathlib
import re
import pandas as pd
import numpy as np
from typing import Union
from tTEM_toolbox.defaults.constants import XYZ_FILE_PATTERN, DOI_FILE_PATTERN


class ProcessTTEM:
    """
    This function is used to format the tTEM data, and return a dataframe that contains filtered/processed tTEM data.\n
    if the input is a string or pathlib.PurePath object, the function will read the tTEM data from the file and \
    return a dataframe.\n
    if the input is a dataframe, the function will reuse the dataframe and return a dataframe.\n
    if the input is a list, the function will read all the tTEM data from the files in the list and return a \
    dataframe.\n
    Version 11.18.2023 \n
    :param fname: A string or pathlib.PurePath object that contains the path to the tTEM .xyz file exported from Aarhus Workbench
    :param doi_path: A string or pathlib.PurePath object that contains the path to the DOI file exported from Aarhus Workbench
    :param layer_exclude: A list that contains the layer number that you want to exclude from the tTEM data
    :param line_exclude: A list that contains the line number that you want to exclude from the tTEM data
    :param point_exclude: A list that contains the point number that you want to exclude from the tTEM data
    :param resample: A int value that indicates whether to fill the tTEM data with a factor, defaults is False
    :return: A pandas dataframe that contains the filtered/processed tTEM data
    """
    def __init__(self,
                 fname: Union[pathlib.PurePath, str, pd.DataFrame,list],
                 doi_path: Union[pathlib.PurePath, str, list] = False,
                 layer_exclude: list = False,
                 line_exclude: list = False,
                 point_exclude: list = False,
                 resample: int = False):
        if not isinstance(fname, list):
            fname = [fname]
        if not isinstance(doi_path, list) and doi_path:
            doi_path = [doi_path]
        self.fname = fname
        self.doi_path = doi_path
        self.layer_exclude = layer_exclude
        self.line_exclude = line_exclude
        self.point_exclude = point_exclude
        self.resample = resample
        self.ttem_data = self.format_ttem()


    @staticmethod
    def _read_ttem(fname: Union[pathlib.PurePath, str]) -> Union[pd.DataFrame, dict]:
        """
        This function read tTEM data from .xyz file, and return a formatted dataframe that contains all the tTEM data. \n
        Version 11.18.2023 \n
        :param fname: A string or pathlib.PurePath object that contains the path to the tTEM .xyz file exported from Aarhus Workbench
        :return: A pandas dataframe that contains all the tTEM data without any filtering
        """
        with open(fname, 'r') as file:
            lines = file.readlines()
        regex = re.compile(XYZ_FILE_PATTERN)
        match_index = []
        for index, line in enumerate(lines):
            if regex.search(line):
                match_index.append(index)
        if len(match_index) == 0:
            raise ValueError("No tTEM data found in the {} !".format(fname))
        elif len(match_index) > 1:
            raise ValueError("There might be duplicate tTEM data in {}, multiple column names found!".format(fname))
        data = [line[1::].strip().split() for line in lines[match_index[0]::]]
        df = pd.DataFrame(data[1::], columns=data[0])
        df = df.astype({'ID': 'int64',
                                'Line_No': 'int64',
                                'Layer_No': 'int64',
                                'UTMX': 'float64',
                                'UTMY': 'float64',
                                'Elevation_Cell': 'float64',
                                'Resistivity': 'float64',
                                'Resistivity_STD': 'float64',
                                'Conductivity': 'float64',
                                'Depth_top': 'float64',
                                'Depth_bottom': 'float64',
                                'Thickness': 'float64',
                                'Thickness_STD': 'float64'
                                })
        df = df[~(df['Thickness_STD'] == float(9999))]
        return df

    @staticmethod
    def _DOI(dataframe: pd.DataFrame,
             doi_path: Union[pathlib.PurePath, str, list]) -> pd.DataFrame:
        """
        Remove all tTEM data under DOI elevation limit with provided DOI file from Aarhus Workbench \n
        Version 11.18.2023 \n
        :param dataframe: Datafram that constains tTEM data
        :param doi_path: path-like contains DOI file, or a list of path that contains multiple DOI files
        :return: Filtered tTEM data above DOI
        """
        from pathlib import Path
        doi_concatlist = []
        match_index = []
        regex = re.compile(DOI_FILE_PATTERN)
        for i in doi_path:
            with open(i, 'r') as file:
                lines = file.readlines()
                print('Applying DOI from {}'.format(Path(i).name))
                for index, line in enumerate(lines):
                    if regex.search(line):
                        match_index.append(index)
            if len(match_index) == 0:
                raise ValueError("No DOI data found in the {}!".format(doi_path))
            elif len(match_index) > 1:
                raise ValueError("There might be duplicate DOI data in {}, multiple column names found!".format(doi_path))
            data = [line[1::].strip().split() for line in lines[match_index[0]::]]
            tmp_doi_df = pd.DataFrame(data[1::], columns=data[0])
            doi_concatlist.append(tmp_doi_df)
        df_DOI = pd.concat(doi_concatlist)
        df_DOI = df_DOI.astype({'UTMX': 'float64',
                                'UTMY': 'float64',
                                'Value': 'float64'
                                })
        df_group = dataframe.groupby(['UTMX', 'UTMY'])
        ttem_concatlist = []
        for name, group in df_group:
            elevation = df_DOI.loc[(df_DOI['UTMX'] == name[0]) & (df_DOI['UTMY'] == name[1])]['Value'].values[0]
            if elevation is None:
                continue
            else:
                new_group = group[group['Elevation_Cell'] >= elevation]
                ttem_concatlist.append(new_group)
        df_out = pd.concat(ttem_concatlist)
        df_out["Elevation_End"] = df_out["Elevation_Cell"].subtract(df_out["Thickness"]) #create new column for future usage
        return df_out

    @staticmethod
    def _layer_exclude(dataframe: pd.DataFrame,
                       layer_exclude: list) -> pd.DataFrame:
        df_out = dataframe[~np.isin(dataframe["Layer_No"], layer_exclude)]
        print('Exclude layer {}'.format(layer_exclude))
        return df_out

    @staticmethod
    def _line_exclude(dataframe: pd.DataFrame,
                      line_exclude: list) -> pd.DataFrame:
        df_out = dataframe[~np.isin(dataframe["Line_No"], line_exclude)]
        print('Exclude line {}'.format(line_exclude))
        return df_out

    @staticmethod
    def _point_exclude(dataframe: pd.DataFrame,
                       point_exclude: list) -> pd.DataFrame:
        df_out = dataframe[~dataframe[['UTMX','UTMY']].isin(point_exclude)]
        [print('Exclude point {},{}'.format(x[0], x[1])) for x in point_exclude]
        return df_out

    @staticmethod
    def _to_linear(group: pd.DataFrame,
                   factor: int) -> pd.DataFrame:
        """
        The core algorithm of the resample method, it fills the tTEM from log to linear.\n
        Version 11.18.2023\n
        :param group: tTEM dataframe, typically a groups from pd.groupby method
        :param factor: how thin your thickness should be divided, e.g. 10 means 1/10 m thickness
        :return: linear thickness tTEM dataframe
        """

        newgroup = group.loc[group.index.repeat(group.Thickness * factor)]
        mul_per_gr = newgroup.groupby('Elevation_Cell').cumcount()
        newgroup['Elevation_Cell'] = newgroup['Elevation_Cell'].subtract(mul_per_gr * 1 / factor)
        newgroup['Depth_top'] = newgroup['Depth_top'].add(mul_per_gr * 1 / factor)
        newgroup['Depth_bottom'] = newgroup['Depth_top'].add(1 / factor)
        newgroup['Elevation_End'] = newgroup['Elevation_Cell'].subtract(1 / factor)
        newgroup['Thickness'] = 1 / factor
        return newgroup

    @staticmethod
    def _resample(dataframe: pd.DataFrame,
                  factor: int) -> pd.DataFrame:
        """
        This staticmethod is connected with format_ttem method, it converts the tTEM thickness from log to linear \
        layers to constant thickness layers.\n
        Version 11.18.2023\n
        :param dataframe: Dataframe that contains the tTEM data
        :param factor: how thin your thickness should be divided, e.g. 10 means 1/10 m thickness
        :return: resampled dataframe
        """
        concatlist = []
        groups = dataframe.groupby(['UTMX', 'UTMY'])
        for name, group in groups:
            newgroup = ProcessTTEM._to_linear(group, factor)
            concatlist.append(newgroup)
        result = pd.concat(concatlist)
        result.reset_index(drop=True, inplace=True)
        return result
    def data(self):
        return self.ttem_data
    def format_ttem(self):
        """
        This is the core method of the class that read file under varies input circumstances, and return a \
        formatted dataframe that contains filtered tTEM data. \n
        Version: 11.18.2023\n
        :return: A pandas dataframe that contains filtered tTEM data
        """
    # Read data under different input circumstances
        from pathlib import Path
        tmp_df = pd.DataFrame()
        if len(self.fname) == 0:
            raise ValueError("The input is empty!")
        if isinstance(self.fname[0], (str, pathlib.PurePath)):
            concatlist = []
            for i in self.fname:
                tmp_df = self._read_ttem(i)
                concatlist.append(tmp_df)
                print("Reading data from file {}...".format(Path(i).name))
            tmp_df = pd.concat(concatlist)
        elif isinstance(self.fname[0], pd.DataFrame):
            print("Reading data from cache...")
            tmp_df = pd.concat(self.fname)
        if tmp_df.empty:
            raise ValueError("The input is empty!")
    # Create filter parameters
        if self.layer_exclude:
            tmp_df = self._layer_exclude(tmp_df, self.layer_exclude)
        if self.line_exclude:
            tmp_df = self._layer_exclude(tmp_df, self.line_exclude)
        if self.point_exclude:
            tmp_df = self._point_exclude(tmp_df, self.point_exclude)
        if self.doi_path:
            tmp_df = self._DOI(tmp_df, self.doi_path)
        if self.resample:
            tmp_df = self._resample(tmp_df, self.resample)
    # Sort the dataframe
        tmp_df = tmp_df.sort_values(by=['ID', 'Line_No','Layer_No'])
        tmp_df.reset_index(drop=True, inplace=True)
        self.ttem_data = tmp_df.copy()
        return self.ttem_data


if __name__ == "__main__":
    print('This is a module, please import it to use it.')
    import tTEM_toolbox
    from pathlib import Path
    workdir = Path.cwd()
    ttem_lslake = workdir.parent.parent.joinpath(r'data\PD22_I03_MOD.xyz')
    ttem_lsl = tTEM_toolbox.ProcessTTEM(ttem_lslake)
