    def fit_diamond_peaks_whittaker(self):
            """ Fits a diamond spectrum to an ideal spectrum accounting for saturated peaks. Spectrum needs to be interpolated to the same spacing as the typeIIA diamond spectrum.  

            Args:
                ideal_diamond (_type_, optional): _description_. Defaults to typeIIA_Spectrum.
            """

            ideal_diamond_Y = self.interpolated_typeIIA_Spectrum.Y
            fit_mask_idx = self.test_diamond_saturation()
            X = self.X
            Y = self.median_filter(11).Y
            def baseline_diamond_fit_R_squared(baseline_input_tuple, spectrum_wavenumber = X ,spectrum_intensity = Y, 
                                               typeIIA_intensity=ideal_diamond_Y, mask_idx_list=fit_mask_idx):
                lam, p = baseline_input_tuple
                lam = np.round(lam,3)
                print(f"lam = {lam}, p = {p}")

                baseline = pybl.whittaker.asls(spectrum_intensity, lam=lam, p=p)[0]

                baseline_subtracted = spectrum_intensity - baseline 
                baseline_subtracted_masked = baseline_subtracted[mask_idx_list]
                typeIIA_masked = typeIIA_intensity[mask_idx_list]
                fit_ratio =  baseline_subtracted_masked/ typeIIA_masked
                
                # Force Baseline to fit flat part of spectrum
                flat_range_idx = (spectrum_wavenumber > 4000) & (spectrum_wavenumber < 5000)
                weight_factor = 0.0001 # Sets balance of residuals between typeIIA and flat baseline section
                flat_baseline_residuals_squared = ((baseline_subtracted[flat_range_idx])**2).sum() * weight_factor 

                # Residuals for the TypeIIA Spectrum and the baseline subtracted data. 
                typeIIa_residuals_squared = (( (baseline_subtracted_masked/fit_ratio) - typeIIA_masked)**2).sum() 

                Total_residuals_squares = flat_baseline_residuals_squared + typeIIa_residuals_squared
                print(f" total Residuals squared {Total_residuals_squares}")
                #return np.log(Total_residuals_squares)
                return Total_residuals_squares
            

            p_opt = optimize.differential_evolution(baseline_diamond_fit_R_squared, bounds=((1e7, 1e10), (1e-7,0.001)), x0=(10000000,0.0005), tol = 1000000000, atol = 100000)
            baseline_opt = pybl.whittaker.asls(self.median_filter(11).Y, lam=p_opt.x[0], p=p_opt.x[1])[0]
        
            return baseline_opt



    def fit_diamond_peaks(self, baseline_algorithm = "Whittaker"):
        """ Fits a diamond spectrum to an ideal spectrum accounting for saturated peaks. Spectrum needs to be interpolated to the same spacing as the typeIIA diamond spectrum.  

        Args:
            ideal_diamond (_type_,
              optional): _description_. Defaults to typeIIA_Spectrum.
        """
        ideal_diamond = self.interpolated_typeIIA_Spectrum
        fit_mask_idx = self.test_diamond_saturation()
        baseline_func = select_baseline_func(self, baseline_algorithm)

        def baseline_diamond_fit_R_squared(baseline_input_tuple, spectrum_wavenumber = self.X ,spectrum_intensity = self.median_filter(11).Y, typeIIA_intensity=ideal_diamond.Y, mask_idx_list=fit_mask_idx):
            lam, p = baseline_input_tuple
            print(f"lam = {lam}, p = {p}")
            
            baseline = baseline_func(spectrum_intensity, lam=lam, p=p)


            baseline_subtracted = spectrum_intensity - baseline 
            baseline_subtracted_masked = baseline_subtracted[mask_idx_list]
            typeIIA_masked = typeIIA_intensity[mask_idx_list]
            fit_ratio =  baseline_subtracted_masked/ typeIIA_masked
            
            # Force Baseline to fit flat part of spectrum
            flat_range_idx = (spectrum_wavenumber > 4000) & (spectrum_wavenumber < 5000)
            weight_factor = 0.0001 # Sets balance of residuals between typeIIA and flat baseline section
            flat_baseline_residuals_squared = ((baseline_subtracted[flat_range_idx])**2).sum() * weight_factor 

            typeIIa_residuals_squared = (( (baseline_subtracted_masked/fit_ratio) - typeIIA_masked)**2).sum() 

            Total_residuals_squares = flat_baseline_residuals_squared + typeIIa_residuals_squared
            print(f" total Residuals squared {Total_residuals_squares}")
            #return np.log(Total_residuals_squares)
            return Total_residuals_squares
        
        p_opt = optimize.differential_evolution(baseline_diamond_fit_R_squared, bounds=((1e7, 1e10), (1e-7,0.001)), x0=(10000000,0.0005), tol = 1000)
        baseline_opt = baseline_als(y=self.median_filter(11).Y , lam=p_opt.x[0], p=p_opt.x[1])
        
        return baseline_opt


    def select_baseline_func(self, baseline_algorithm = "Whittaker"):
        """ Fits a diamond spectrum to an ideal spectrum accounting for saturated peaks. Spectrum needs to be interpolated to the same spacing as the typeIIA diamond spectrum.  

        Args:
            ideal_diamond (_type_,
              optional): _description_. Defaults to typeIIA_Spectrum.
        """
        def baseline_Whittaker_internal(spectrum_intensity, lam=lam, p=p):
            return pybl.whittaker.asls(spectrum_intensity, lam=lam, p=p)[0]

        match baseline_algorithm:
            case "Whittaker":
                baseline_func = baseline_Whittaker_internal
            case "ALS":

                baseline_func = baseline_als
            case _:
                print("Incorrect Baseline Option Selected")

        return baseline_func


        