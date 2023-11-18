import numpy as np
import pandas as pd
import xarray as xr
import pathlib
#import concurrent.

class ProcessTTEM():
    def __init__(self, ttem_path, **kwargs):
        self.ttem_path = ttem_path
        self.kwargs = kwargs
        ttem = core.process_ttem.format_ttem(fname=self.ttem_path, **self.kwargs)
        self.ttem = ttem
    def data(self):
        return self.ttem
    def fill(self, factor=100):
        ttem_fill = core.process_ttem.format_ttem(fname=self.ttem, filling=True, factor=factor)
        return ttem_fill
    def ttem_well_connect(self, distance=500, wellupscale=100, check_corr=np.nan, debug=False):
        if np.isin("welllog", list(self.kwargs.keys())):
            if isinstance(self.kwargs['welllog'], (str, pathlib.PurePath)):
                welllog_upscale = core.process_well.format_well(self.kwargs["welllog"], upscale=wellupscale)
            elif isinstance(self.kwargs['welllog'], pd.DataFrame):
                welllog_upscale = self.kwargs['welllog']
            if ~np.isnan(check_corr):
                ttem_match, well_match = core.bootstrap.select_closest(self.ttem, welllog_upscale,
                                                                       distance=distance, showskip=False)
                welllog_upscale = core.bootstrap.corr_well_filter(ttem_match, well_match, corr_thershold=check_corr)
                pre_bootstrap, Resistivity, Thickness_ratio,matched_ttem, matched_well = core.bootstrap.pre_bootstrap(self.ttem, welllog_upscale,
                                                                                                                      distance=distance)
            else:
                pre_bootstrap, Resistivity, Thickness_ratio,matched_ttem, matched_well = core.bootstrap.pre_bootstrap(self.ttem, welllog_upscale,
                                                                                                                      distance=distance)
            fine, mix, coarse = core.bootstrap.bootstrap(Resistivity, Thickness_ratio)
            pack_bootstrap_result = pd.DataFrame({'fine':fine,'mix':mix,'coarse':coarse})
            Resi_conf_df = core.bootstrap.packup(fine, mix, coarse)
            rk_trans= core.Rock_trans.rock_transform(self.ttem, Resi_conf_df)
        else:
            raise ValueError("welllog not found!")
        if debug is True:
            return pre_bootstrap,rk_trans,Resistivity, Thickness_ratio, pack_bootstrap_result, Resi_conf_df, matched_ttem, matched_well
        else:
            return rk_trans, pack_bootstrap_result
class ProcessGamma():
    def __init__(self, gamma, gammaloc, **kwargs):
        self.gamma = gamma
        self.gammaloc = gammaloc
        self.kwargs = kwargs
    def process(self):
        if np.isin('columns', list(self.kwargs.keys())):
            ori = core.process_gamma.load_gamma(self.gamma, columns=self.kwargs["columns"])
        else:
            ori = core.process_gamma.load_gamma(self.gamma)
        ori = core.process_gamma.georeference(ori, self.gammaloc)
        if np.isin("rolling", list(self.kwargs.keys())):
            if self.kwargs["rolling"] is True:
                try:
                    ori = core.process_gamma.rolling_average(ori,
                                                             window=self.kwargs["window"])
                except:
                    ori = core.process_gamma.rolling_average(ori)
        return ori
    def gr_wlg_combine(self,ori):
        try:
            gr_wlg = core.process_gamma.gamma_well_connect(ori, self.kwargs["welllog"])
        except:
            raise("Process gamma data first use .process()")

        return gr_wlg

    def gr_ttem_combine(self, ori, **kwargs):
        ttem_result = pd.DataFrame()
        if np.isin("ttem", list(kwargs.keys())):
            try:
                ttem_result = kwargs["ttem"]
            except:
                try:
                    ttem_result = ProcessTTEM(ttemname=kwargs["ttem"],
                                                    welllog=kwargs['welllog'],
                                                    DOI=kwargs['DOI']).data()
                except:
                    raise ValueError("\nMissing parameters")
        try:
            ori_result = ori
        except:
            raise ValueError("Process gamma data first use .process()")
        gr_ttem = core.process_gamma.gamma_ttem_connect(ori_result, ttem_result)
        return gr_ttem
class GWSurface():
    """
    Parameters
    ----------
    waterwell : could be a single well or path to a file that contains a list of well name.
    elevation_type: Choose to keep which type of data could be 'NAVD88', 'NGVD29', or ''depth'
    time: Select a time period to export, could be a single time or a list of time, format YYYY-MM-DD
    """
    def __init__(self, waterwell,
                 elevation_type='NAVD88',
                 *args, **kwargs):
        self.well = waterwell
        self.elevation_type = elevation_type
        self.args = args
        self.df = pd.DataFrame()
        self.kwargs = kwargs
        if isinstance(self.well, int):
            print('reading wells in file {}'.format(self.well))
            self.well = pd.read_excel(self.well)
            self.well_list = self.well['SiteNumber'].values
        elif isinstance(self.well, (str, pathlib.PurePath)):
            try:
                self.well = pd.read_excel(self.well)
            except:
                try:
                    self.well= pd.read_csv(self.well)
                except:
                    print('{} is not in xls or xlsx format try read as csv'.format(self.well))
            self.well_list = self.well['SiteNumber'].values
        self.ds = xr.Dataset()
        for i in self.well_list:
            tmp_ds = core.process_well.format_usgs_water(str(i), self.elevation_type, **self.kwargs)
            if tmp_ds is None:
                print('{} does not have water level data'.format(i))
                continue
            try:
                self.ds = self.ds.merge(tmp_ds)
                # ds = ds.merge(tmp_ds)
            except:
                print('{} not able to merge, try to solve the problem by drop duplicates.'.format(str(i)))
                try:
                    self.ds = self.ds.merge(tmp_ds[str(i)].drop_duplicates(dim='time').to_dataset())
                except:
                    print('{} failed to merge'.format(str(i)))
        print('All Wells Done!')
    def data(self):
        """
        :return: returns a xarray dataset include formatted USGS water well information
        """
        return self.ds
    def format(self, elevation=None, time='2022-03'):
        if self.elevation_type in ['NAVD88', 'NGVD29']:
            header = 'sl_lev_va'
        elif self.elevation_type.lower() == 'depth':
            header = 'lev_va'
        else:
            raise ValueError('{} not one of NAVD88, NGVD29, or depth'.format(self.elevation_type))
        if isinstance(time, str):
            self.df = core.process_well.water_head_format(self.ds, time=str(time), header=header, elevation=elevation)
        else:
            self.df = core.process_well.water_head_format(self.ds, time=str(time[0]), header=header, elevation=elevation)
            for i in time[1:]:
                tmp_df = core.process_well.water_head_format(self.ds, time=str(i), header=header, elevation=elevation)
                self.df = self.df.merge(tmp_df, how='left', on=['wellname','lat','long','datum','UTMX','UTMY','well_depth','altitude'])
        return self.df


#df.sl_lev_va=df.sl_lev_va.astype(float).div(3.2808)








