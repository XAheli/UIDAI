"""
Pincode API Client
Author: Shuvam Banerji Seal's Team
UIDAI Hackathon - Aadhaar Data Analysis

Provides pincode information from India Post API:
- Pincode details
- Post office information
- District/State mapping
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from .base import BaseAPIClient

logger = logging.getLogger(__name__)


class PincodeClient(BaseAPIClient):
    """
    Client for India Post Pincode API
    Free, no API key required
    """
    
    def __init__(
        self,
        rate_limit: float = 5.0,
        cache_ttl: int = 2592000  # 30 days (pincode data rarely changes)
    ):
        """
        Initialize pincode client
        
        Args:
            rate_limit: Requests per second
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__(
            base_url='https://api.postalpincode.in',
            api_key=None,
            rate_limit=rate_limit,
            cache_ttl=cache_ttl
        )
    
    def health_check(self) -> bool:
        """Check if API is available"""
        try:
            result = self._make_request('pincode/110001')
            return result.get('Status') == 'Success'
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def get_pincode_details(
        self,
        pincode: Union[str, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific pincode
        
        Args:
            pincode: 6-digit Indian pincode
            
        Returns:
            Dict with pincode details
        """
        pincode = str(pincode).zfill(6)
        
        if not pincode.isdigit() or len(pincode) != 6:
            logger.warning(f"Invalid pincode format: {pincode}")
            return None
        
        try:
            result = self._make_request(f'pincode/{pincode}')
            
            if result.get('Status') != 'Success':
                return None
            
            post_offices = result.get('PostOffice', [])
            if not post_offices:
                return None
            
            # Use first post office for primary details
            primary = post_offices[0]
            
            return {
                'pincode': pincode,
                'name': primary.get('Name', ''),
                'district': primary.get('District', ''),
                'state': primary.get('State', ''),
                'division': primary.get('Division', ''),
                'region': primary.get('Region', ''),
                'circle': primary.get('Circle', ''),
                'branch_type': primary.get('BranchType', ''),
                'delivery_status': primary.get('DeliveryStatus', ''),
                'post_offices': [
                    {
                        'name': po.get('Name', ''),
                        'branch_type': po.get('BranchType', ''),
                        'delivery': po.get('DeliveryStatus', '')
                    }
                    for po in post_offices
                ],
                'post_office_count': len(post_offices)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get details for pincode {pincode}: {str(e)}")
            return None
    
    def search_by_postoffice(
        self,
        postoffice_name: str
    ) -> List[Dict[str, Any]]:
        """
        Search pincodes by post office name
        
        Args:
            postoffice_name: Name of post office
            
        Returns:
            List of matching pincodes
        """
        try:
            result = self._make_request(f'postoffice/{postoffice_name}')
            
            if result.get('Status') != 'Success':
                return []
            
            return [
                {
                    'pincode': po.get('Pincode', ''),
                    'name': po.get('Name', ''),
                    'district': po.get('District', ''),
                    'state': po.get('State', ''),
                    'branch_type': po.get('BranchType', '')
                }
                for po in result.get('PostOffice', [])
            ]
            
        except Exception as e:
            logger.warning(f"Search failed for '{postoffice_name}': {str(e)}")
            return []
    
    def get_batch_pincode_details(
        self,
        pincodes: List[Union[str, int]]
    ) -> pd.DataFrame:
        """
        Get details for multiple pincodes
        
        Args:
            pincodes: List of pincodes
            
        Returns:
            DataFrame with pincode details
        """
        results = []
        
        for i, pincode in enumerate(pincodes):
            details = self.get_pincode_details(pincode)
            if details:
                results.append(details)
            else:
                results.append({
                    'pincode': str(pincode).zfill(6),
                    'error': 'Not found'
                })
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(pincodes)} pincodes")
        
        return pd.DataFrame(results)
    
    def get_state_pincodes(
        self,
        state: str
    ) -> List[Dict[str, Any]]:
        """
        Get all pincodes for a state (limited)
        Note: API doesn't support this directly, using local knowledge
        
        Args:
            state: State name
            
        Returns:
            List of pincodes in the state
        """
        # Pincode ranges by state (approximate first 2-3 digits)
        state_prefixes = {
            'DELHI': ['110'],
            'UTTAR PRADESH': ['201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '262', '263', '271', '272', '273', '274', '275', '276', '277', '281', '282', '283', '284', '285'],
            'MAHARASHTRA': ['400', '401', '402', '403', '410', '411', '412', '413', '414', '415', '416', '421', '422', '423', '424', '425', '431', '440', '441', '442', '443', '444', '445'],
            'KARNATAKA': ['560', '561', '562', '563', '570', '571', '572', '573', '574', '575', '576', '577', '580', '581', '582', '583', '584', '585', '586', '587', '590', '591'],
            'TAMIL NADU': ['600', '601', '602', '603', '604', '605', '606', '607', '608', '609', '610', '611', '612', '613', '614', '620', '621', '622', '623', '624', '625', '626', '627', '628', '629', '630', '631', '632', '635', '636', '637', '638', '639', '641', '642', '643'],
            'WEST BENGAL': ['700', '711', '712', '713', '721', '722', '723', '731', '732', '733', '734', '735', '736', '741', '742', '743'],
            'RAJASTHAN': ['301', '302', '303', '304', '305', '306', '307', '311', '312', '313', '314', '321', '322', '323', '324', '325', '326', '327', '328', '331', '332', '333', '334', '335', '341', '342', '343', '344', '345'],
            'GUJARAT': ['360', '361', '362', '363', '364', '365', '370', '380', '382', '383', '384', '385', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396'],
            'ANDHRA PRADESH': ['515', '516', '517', '518', '519', '520', '521', '522', '523', '524', '530', '531', '532', '533', '534', '535'],
            'TELANGANA': ['500', '501', '502', '503', '504', '505', '506', '507', '508', '509'],
            'KERALA': ['670', '671', '672', '673', '674', '675', '676', '677', '678', '679', '680', '681', '682', '683', '685', '686', '688', '689', '690', '691', '692', '693', '695'],
            'MADHYA PRADESH': ['450', '451', '452', '453', '454', '455', '456', '457', '458', '460', '461', '462', '463', '464', '465', '466', '467', '470', '471', '472', '473', '474', '475', '476', '477', '480', '481', '482', '483', '484', '485', '486', '487', '488'],
            'BIHAR': ['800', '801', '802', '803', '804', '805', '806', '811', '812', '813', '821', '822', '823', '824', '831', '841', '842', '843', '844', '845', '846', '847', '848', '851', '852', '853', '854', '855'],
            'ODISHA': ['751', '752', '753', '754', '755', '756', '757', '758', '759', '760', '761', '762', '763', '764', '765', '766', '767', '768', '769', '770'],
            'PUNJAB': ['140', '141', '142', '143', '144', '145', '146', '147', '148', '150', '151', '152', '160'],
            'HARYANA': ['121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136'],
            'ASSAM': ['781', '782', '783', '784', '785', '786', '787', '788'],
        }
        
        state_upper = state.upper()
        prefixes = state_prefixes.get(state_upper, [])
        
        if not prefixes:
            logger.warning(f"No pincode prefixes known for state: {state}")
            return []
        
        return [{
            'state': state,
            'pincode_prefixes': prefixes,
            'note': 'Use these prefixes to construct full pincodes'
        }]
    
    def validate_pincode_state(
        self,
        pincode: Union[str, int],
        expected_state: str
    ) -> Dict[str, Any]:
        """
        Validate if a pincode belongs to expected state
        
        Args:
            pincode: 6-digit pincode
            expected_state: Expected state name
            
        Returns:
            Validation result
        """
        details = self.get_pincode_details(pincode)
        
        if not details:
            return {
                'pincode': str(pincode).zfill(6),
                'valid': False,
                'error': 'Pincode not found'
            }
        
        actual_state = details.get('state', '').upper()
        expected_upper = expected_state.upper()
        
        # Handle common variations
        state_aliases = {
            'JAMMU AND KASHMIR': ['JAMMU & KASHMIR', 'J&K'],
            'ANDAMAN AND NICOBAR ISLANDS': ['ANDAMAN AND NICOBAR', 'A&N ISLANDS'],
            'DADRA AND NAGAR HAVELI AND DAMAN AND DIU': ['DADRA NAGAR HAVELI', 'DAMAN AND DIU', 'DNH AND DD']
        }
        
        is_match = actual_state == expected_upper
        if not is_match:
            for canonical, aliases in state_aliases.items():
                if expected_upper in [canonical] + aliases and actual_state in [canonical] + aliases:
                    is_match = True
                    break
        
        return {
            'pincode': str(pincode).zfill(6),
            'valid': is_match,
            'expected_state': expected_state,
            'actual_state': details.get('state', ''),
            'district': details.get('district', '')
        }


__all__ = ['PincodeClient']
