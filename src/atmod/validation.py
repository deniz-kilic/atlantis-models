"""
Validation module for Atlantis subsurface models.

This module provides comprehensive validation of Atlantis model NetCDF files
to ensure they meet all requirements for Atlantis Julia subsidence modeling
and optionally for ArcGIS compatibility.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue found in a model.

    Attributes
    ----------
    severity : str
        'ERROR', 'WARNING', or 'INFO'
    category : str
        Category of issue: 'metadata', 'data_range', 'spatial', 'structure', 'cf_compliance'
    variable : str
        Name of the variable related to this issue
    message : str
        Human-readable description of the issue
    details : Dict, optional
        Additional details about the issue (e.g., actual values, expected values)
    """
    severity: str  # 'ERROR', 'WARNING', 'INFO'
    category: str  # 'metadata', 'data_range', 'spatial', 'structure', 'cf_compliance'
    variable: str
    message: str
    details: Optional[Dict] = None


@dataclass
class ValidationReport:
    """
    Container for validation results.

    Provides methods to add issues, query by severity, and generate summaries.
    """
    issues: List[ValidationIssue] = field(default_factory=list)

    def add(self, issue: ValidationIssue):
        """Add a validation issue to the report."""
        self.issues.append(issue)

    def add_multiple(self, issues: List[ValidationIssue]):
        """Add multiple validation issues."""
        self.issues.extend(issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get all ERROR severity issues."""
        return [i for i in self.issues if i.severity == 'ERROR']

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get all WARNING severity issues."""
        return [i for i in self.issues if i.severity == 'WARNING']

    @property
    def infos(self) -> List[ValidationIssue]:
        """Get all INFO severity issues."""
        return [i for i in self.issues if i.severity == 'INFO']

    @property
    def is_valid(self) -> bool:
        """Check if model has no errors (warnings are acceptable)."""
        return len(self.errors) == 0

    def print_summary(self, show_details=False):
        """
        Print validation report summary.

        Parameters
        ----------
        show_details : bool
            If True, print all issues with details
        """
        print("\n" + "=" * 70)
        print("VALIDATION REPORT SUMMARY")
        print("=" * 70)
        print(f"Errors:   {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Info:     {len(self.infos)}")
        print(f"\nOverall Status: {'✓ VALID' if self.is_valid else '✗ INVALID'}")

        if self.errors and show_details:
            print("\n" + "-" * 70)
            print("ERRORS:")
            print("-" * 70)
            for issue in self.errors:
                print(f"\n[{issue.variable}] {issue.category}")
                print(f"  {issue.message}")
                if issue.details:
                    print(f"  Details: {issue.details}")

        if self.warnings and show_details:
            print("\n" + "-" * 70)
            print("WARNINGS:")
            print("-" * 70)
            for issue in self.warnings:
                print(f"\n[{issue.variable}] {issue.category}")
                print(f"  {issue.message}")
                if issue.details:
                    print(f"  Details: {issue.details}")

        print("=" * 70)

    def to_dict(self) -> Dict:
        """Convert report to dictionary for serialization."""
        return {
            'n_errors': len(self.errors),
            'n_warnings': len(self.warnings),
            'n_info': len(self.infos),
            'is_valid': self.is_valid,
            'issues': [
                {
                    'severity': i.severity,
                    'category': i.category,
                    'variable': i.variable,
                    'message': i.message,
                    'details': i.details
                }
                for i in self.issues
            ]
        }


class AtlantisModelValidator:
    """
    Comprehensive validator for Atlantis subsurface models.

    This validator checks:
    - Required variables presence
    - Dimension structure
    - Metadata completeness (units, long_name, etc.)
    - Data value ranges (physical validity)
    - Spatial consistency between variables
    - CF-1.8 compliance (optional)
    - ArcGIS compatibility (optional)
    """

    # Required variables for Atlantis Julia
    REQUIRED_VARS_3D = [
        'lithology',
        'thickness',
        'geology',
        'mass_fraction_organic',
        'rho_bulk'
    ]

    REQUIRED_VARS_2D = [
        'surface_level',
        'phreatic_level',
        'zbase',
        'domainbase',
        'max_oxidation_depth'
    ]

    OPTIONAL_VARS_2D = [
        'no_oxidation_thickness',
        'no_shrinkage_thickness',
        'aquifer_head'
    ]

    OPTIONAL_VARS_3D = [
        'mass_fraction_lutum',
        'shrinkage_degree'
    ]

    # Expected data ranges for physical validity
    DATA_RANGES = {
        'thickness': (0.0, 100.0),  # meters
        'rho_bulk': (100.0, 3000.0),  # kg/m³
        'mass_fraction_organic': (0.0, 1.0),  # fraction
        'mass_fraction_lutum': (0.0, 1.0),  # fraction
        'shrinkage_degree': (0.0, 1.0),  # fraction
        'surface_level': (-50.0, 500.0),  # m NAP (Dutch elevation)
        'phreatic_level': (-1000.0, 500.0),  # m NAP
        'zbase': (-100.0, 0.0),  # m NAP (model base)
        'max_oxidation_depth': (0.0, 10.0),  # meters
        'lithology': (0, 10),  # lithology classes
        'geology': (0, 10),  # geology classes
    }

    # Required attributes per variable
    REQUIRED_ATTRS = {
        'all': ['units', 'long_name'],  # Required for all variables
        'coordinates': ['standard_name', 'axis'],  # Required for coordinate variables
    }

    def __init__(self, ds: xr.Dataset):
        """
        Initialize validator with a dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Atlantis model dataset to validate
        """
        self.ds = ds
        self.report = ValidationReport()

    def validate_all(self, check_arcgis=False) -> ValidationReport:
        """
        Run all validation checks.

        Parameters
        ----------
        check_arcgis : bool
            If True, also check ArcGIS compatibility requirements

        Returns
        -------
        ValidationReport
            Complete validation report
        """
        print("Running Atlantis model validation...")

        self.check_required_variables()
        self.check_dimensions()
        self.check_metadata()
        self.check_data_ranges()
        self.check_spatial_consistency()

        if check_arcgis:
            self.check_arcgis_compatibility()

        print("Validation complete.")
        return self.report

    def check_required_variables(self):
        """Check that all required variables are present."""
        print("  Checking required variables...")

        for var in self.REQUIRED_VARS_3D:
            if var not in self.ds.data_vars:
                self.report.add(ValidationIssue(
                    severity='ERROR',
                    category='structure',
                    variable=var,
                    message=f"Required 3D variable '{var}' is missing"
                ))

        for var in self.REQUIRED_VARS_2D:
            if var not in self.ds.data_vars:
                self.report.add(ValidationIssue(
                    severity='ERROR',
                    category='structure',
                    variable=var,
                    message=f"Required 2D variable '{var}' is missing"
                ))

    def check_dimensions(self):
        """Check dimension structure."""
        print("  Checking dimensions...")

        required_dims = {'layer', 'y', 'x'}
        actual_dims = set(self.ds.dims)

        if not required_dims.issubset(actual_dims):
            missing = required_dims - actual_dims
            self.report.add(ValidationIssue(
                severity='ERROR',
                category='structure',
                variable='dimensions',
                message=f"Missing required dimensions: {missing}"
            ))

        # Check dimension sizes are reasonable
        if 'layer' in self.ds.dims and self.ds.dims['layer'] < 1:
            self.report.add(ValidationIssue(
                severity='ERROR',
                category='structure',
                variable='layer',
                message=f"Layer dimension has invalid size: {self.ds.dims['layer']}"
            ))

        if 'y' in self.ds.dims and self.ds.dims['y'] < 1:
            self.report.add(ValidationIssue(
                severity='ERROR',
                category='structure',
                variable='y',
                message=f"Y dimension has invalid size: {self.ds.dims['y']}"
            ))

        if 'x' in self.ds.dims and self.ds.dims['x'] < 1:
            self.report.add(ValidationIssue(
                severity='ERROR',
                category='structure',
                variable='x',
                message=f"X dimension has invalid size: {self.ds.dims['x']}"
            ))

    def check_metadata(self):
        """Check variable metadata completeness."""
        print("  Checking metadata...")

        required_attrs = self.REQUIRED_ATTRS['all']

        for var in self.ds.data_vars:
            for attr in required_attrs:
                if attr not in self.ds[var].attrs:
                    self.report.add(ValidationIssue(
                        severity='WARNING',
                        category='metadata',
                        variable=var,
                        message=f"Missing '{attr}' attribute"
                    ))

        # Check coordinate variables
        for coord in ['x', 'y', 'layer']:
            if coord in self.ds.coords:
                for attr in self.REQUIRED_ATTRS['coordinates']:
                    if attr not in self.ds[coord].attrs:
                        self.report.add(ValidationIssue(
                            severity='WARNING',
                            category='metadata',
                            variable=coord,
                            message=f"Coordinate '{coord}' missing '{attr}' attribute"
                        ))

    def check_data_ranges(self):
        """Check data values are within expected ranges."""
        print("  Checking data ranges...")

        for var, (min_val, max_val) in self.DATA_RANGES.items():
            if var not in self.ds.data_vars:
                continue

            data = self.ds[var].values
            valid_data = data[~np.isnan(data)]

            if len(valid_data) == 0:
                self.report.add(ValidationIssue(
                    severity='ERROR',
                    category='data_range',
                    variable=var,
                    message="All values are NaN - no valid data"
                ))
                continue

            actual_min = np.nanmin(valid_data)
            actual_max = np.nanmax(valid_data)

            if actual_min < min_val or actual_max > max_val:
                self.report.add(ValidationIssue(
                    severity='WARNING',
                    category='data_range',
                    variable=var,
                    message=f"Values outside expected range [{min_val}, {max_val}]",
                    details={
                        'expected_range': (min_val, max_val),
                        'actual_range': (float(actual_min), float(actual_max))
                    }
                ))

    def check_spatial_consistency(self):
        """Check spatial relationships between variables."""
        print("  Checking spatial consistency...")

        # Check: phreatic level should generally be below or equal to surface
        if 'surface_level' in self.ds and 'phreatic_level' in self.ds:
            surface = self.ds['surface_level'].values
            phreatic = self.ds['phreatic_level'].values

            # Where both are valid and phreatic is not a fill value
            valid_mask = (~np.isnan(surface) & ~np.isnan(phreatic) &
                         (phreatic > -999))  # Exclude common fill value

            if np.any(valid_mask):
                problematic = (phreatic[valid_mask] > surface[valid_mask])
                if np.any(problematic):
                    n_issues = np.sum(problematic)
                    pct_issues = (n_issues / np.sum(valid_mask)) * 100
                    self.report.add(ValidationIssue(
                        severity='WARNING',
                        category='spatial',
                        variable='phreatic_level',
                        message=f"Phreatic level above surface in {n_issues:,} cells ({pct_issues:.1f}%)",
                        details={'n_cells': int(n_issues)}
                    ))

        # Check: thickness should match between layers
        if 'thickness' in self.ds:
            thickness = self.ds['thickness'].values

            # Check for negative thickness
            negative = (thickness < 0) & ~np.isnan(thickness)
            if np.any(negative):
                n_negative = np.sum(negative)
                self.report.add(ValidationIssue(
                    severity='ERROR',
                    category='data_range',
                    variable='thickness',
                    message=f"Found {n_negative:,} cells with negative thickness",
                    details={'n_cells': int(n_negative)}
                ))

    def check_arcgis_compatibility(self):
        """Check ArcGIS compatibility requirements."""
        print("  Checking ArcGIS compatibility...")

        # Check for spatial_ref CRS variable
        if 'spatial_ref' not in self.ds.data_vars and 'crs' not in self.ds.data_vars:
            self.report.add(ValidationIssue(
                severity='INFO',
                category='cf_compliance',
                variable='spatial_ref',
                message="No CRS variable found - ArcGIS may not recognize spatial reference"
            ))

        # Check x, y have standard_name
        for coord in ['x', 'y']:
            if coord in self.ds.coords:
                if 'standard_name' not in self.ds[coord].attrs:
                    self.report.add(ValidationIssue(
                        severity='WARNING',
                        category='cf_compliance',
                        variable=coord,
                        message=f"Coordinate '{coord}' missing 'standard_name' for ArcGIS"
                    ))
                else:
                    expected_name = f'projection_{coord}_coordinate'
                    actual_name = self.ds[coord].attrs['standard_name']
                    if actual_name != expected_name:
                        self.report.add(ValidationIssue(
                            severity='WARNING',
                            category='cf_compliance',
                            variable=coord,
                            message=f"standard_name should be '{expected_name}', found '{actual_name}'"
                        ))

        # Check for level coordinate (for voxel visualization)
        if 'level' not in self.ds:
            self.report.add(ValidationIssue(
                severity='INFO',
                category='cf_compliance',
                variable='level',
                message="Level auxiliary coordinate not present - may limit voxel visualization in ArcGIS"
            ))

        # Check for grid_mapping attribute
        has_grid_mapping = any(
            'grid_mapping' in self.ds[var].attrs
            for var in self.ds.data_vars
            if 'x' in self.ds[var].dims and 'y' in self.ds[var].dims
        )

        if not has_grid_mapping:
            self.report.add(ValidationIssue(
                severity='WARNING',
                category='cf_compliance',
                variable='grid_mapping',
                message="No spatial variables have grid_mapping attribute - ArcGIS may not link to CRS"
            ))


def validate_atlantis_model(netcdf_path: str, check_arcgis: bool = False) -> ValidationReport:
    """
    Validate an Atlantis model NetCDF file.

    Parameters
    ----------
    netcdf_path : str
        Path to the NetCDF file to validate
    check_arcgis : bool
        If True, also check ArcGIS compatibility (default: False)

    Returns
    -------
    ValidationReport
        Validation report with all issues found

    Examples
    --------
    >>> report = validate_atlantis_model('model.nc')
    >>> report.print_summary()
    >>> if report.is_valid:
    ...     print("Model is valid!")
    """
    print(f"Validating: {netcdf_path}")
    print("=" * 70)

    ds = xr.open_dataset(netcdf_path)
    validator = AtlantisModelValidator(ds)
    report = validator.validate_all(check_arcgis=check_arcgis)
    ds.close()

    return report
