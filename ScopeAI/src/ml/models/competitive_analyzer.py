"""
Competitive analysis system for understanding market positioning and company comparisons.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class CompetitivePosition:
    """Competitive position data structure."""
    company_id: str
    company_name: str
    position_x: float
    position_y: float
    cluster_id: int
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]
    similarity_scores: Dict[str, float]
    market_share_estimate: float
    innovation_score: float
    talent_attraction_score: float

class CompetitiveAnalyzer:
    """Advanced competitive analysis and market positioning."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca_model = None
        self.tsne_model = None
        self.clustering_model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.initialized = False
        
    async def initialize(self):
        """Initialize the competitive analyzer."""
        try:
            self.pca_model = PCA(n_components=2)
            self.tsne_model = TSNE(n_components=2, random_state=42, perplexity=5)
            self.clustering_model = KMeans(n_clusters=5, random_state=42)
            
            self.initialized = True
            logger.info("Competitive analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize competitive analyzer: {str(e)}")
            
    async def cleanup(self):
        """Cleanup the competitive analyzer."""
        self.scaler = None
        self.pca_model = None
        self.tsne_model = None
        self.clustering_model = None
        self.tfidf_vectorizer = None
        self.initialized = False
        
    async def analyze_positioning(
        self,
        company_ids: List[str],
        features: pd.DataFrame,
        dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze competitive positioning of companies."""
        
        if not self.initialized:
            await self.initialize()
            
        try:
            logger.info(f"Analyzing competitive positioning for {len(company_ids)} companies")
            
            # Prepare competitive features
            comp_features = await self._prepare_competitive_features(features, company_ids)
            
            if comp_features.empty:
                return {"error": "Insufficient data for competitive analysis"}
                
            # Generate positioning map
            positioning_map = await self._create_positioning_map(comp_features, company_ids)
            
            # Perform clustering analysis
            clusters = await self._perform_clustering_analysis(comp_features, company_ids)
            
            # Analyze strengths and weaknesses
            swot_analysis = await self._perform_swot_analysis(comp_features, company_ids)
            
            # Gap analysis
            gap_analysis = await self._perform_gap_analysis(comp_features, company_ids)
            
            # Strategic recommendations
            recommendations = await self._generate_strategic_recommendations(
                comp_features, positioning_map, clusters, gap_analysis
            )
            
            # Market dynamics
            market_dynamics = await self._analyze_market_dynamics(comp_features, company_ids)
            
            return {
                "positioning_map": positioning_map,
                "clusters": clusters,
                "strengths": swot_analysis["strengths"],
                "gaps": gap_analysis,
                "recommendations": recommendations,
                "dynamics": market_dynamics,
                "analysis_metadata": {
                    "companies_analyzed": len(company_ids),
                    "features_used": list(comp_features.columns),
                    "analysis_date": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing competitive positioning: {str(e)}")
            return {"error": str(e)}
            
    async def compare_companies(
        self,
        company_a_id: str,
        company_b_id: str,
        comparison_dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform detailed comparison between two companies."""
        
        try:
            logger.info(f"Comparing companies {company_a_id} vs {company_b_id}")
            
            # Get company data
            company_a_data = await self._get_company_features(company_a_id)
            company_b_data = await self._get_company_features(company_b_id)
            
            if company_a_data.empty or company_b_data.empty:
                return {"error": "Insufficient data for one or both companies"}
                
            # Dimensional comparison
            dimensional_comparison = await self._compare_dimensions(
                company_a_data, company_b_data, comparison_dimensions
            )
            
            # Similarity analysis
            similarity_analysis = await self._calculate_similarity(company_a_data, company_b_data)
            
            # Competitive advantages
            advantages = await self._identify_competitive_advantages(
                company_a_id, company_b_id, company_a_data, company_b_data
            )
            
            # Talent competition
            talent_competition = await self._analyze_talent_competition(company_a_id, company_b_id)
            
            # Technology overlap
            tech_overlap = await self._analyze_technology_overlap(company_a_id, company_b_id)
            
            return {
                "company_a": company_a_id,
                "company_b": company_b_id,
                "dimensional_comparison": dimensional_comparison,
                "similarity_score": similarity_analysis,
                "competitive_advantages": advantages,
                "talent_competition": talent_competition,
                "technology_overlap": tech_overlap,
                "comparison_summary": await self._generate_comparison_summary(
                    company_a_id, company_b_id, dimensional_comparison, advantages
                )
            }
            
        except Exception as e:
            logger.error(f"Error comparing companies: {str(e)}")
            return {"error": str(e)}
            
    async def identify_market_leaders(
        self,
        industry: Optional[str] = None,
        geographic_region: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Identify market leaders based on various metrics."""
        
        try:
            logger.info("Identifying market leaders")
            
            # Get market data
            market_data = await self._get_market_data(industry, geographic_region)
            
            if market_data.empty:
                return {"leaders": [], "message": "No market data available"}
                
            # Calculate leadership scores
            leadership_scores = await self._calculate_leadership_scores(market_data, metrics)
            
            # Identify leaders by category
            leaders_by_category = await self._categorize_leaders(leadership_scores)
            
            # Analyze leadership trends
            leadership_trends = await self._analyze_leadership_trends(market_data)
            
            # Emerging challengers
            emerging_challengers = await self._identify_emerging_challengers(market_data)
            
            return {
                "leaders": leadership_scores,
                "categories": leaders_by_category,
                "trends": leadership_trends,
                "emerging_challengers": emerging_challengers,
                "market_context": {
                    "industry": industry,
                    "region": geographic_region,
                    "companies_analyzed": len(market_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error identifying market leaders: {str(e)}")
            return {"error": str(e)}
            
    async def analyze_competitive_moats(
        self,
        company_id: str,
        competitor_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze competitive moats and defensibility."""
        
        try:
            logger.info(f"Analyzing competitive moats for {company_id}")
            
            # Get company data
            company_data = await self._get_comprehensive_company_data(company_id)
            
            if company_data.empty:
                return {"error": "Insufficient company data"}
                
            # Analyze different types of moats
            moat_analysis = {}
            
            # Technology moat
            moat_analysis["technology"] = await self._analyze_technology_moat(company_id, company_data)
            
            # Talent moat
            moat_analysis["talent"] = await self._analyze_talent_moat(company_id, company_data)
            
            # Network effects moat
            moat_analysis["network_effects"] = await self._analyze_network_effects_moat(company_id, company_data)
            
            # Brand moat
            moat_analysis["brand"] = await self._analyze_brand_moat(company_id, company_data)
            
            # Data moat
            moat_analysis["data"] = await self._analyze_data_moat(company_id, company_data)
            
            # Regulatory moat
            moat_analysis["regulatory"] = await self._analyze_regulatory_moat(company_id, company_data)
            
            # Overall moat strength
            overall_strength = await self._calculate_overall_moat_strength(moat_analysis)
            
            # Vulnerability analysis
            vulnerabilities = await self._identify_moat_vulnerabilities(moat_analysis, competitor_ids)
            
            return {
                "company_id": company_id,
                "moat_analysis": moat_analysis,
                "overall_strength": overall_strength,
                "vulnerabilities": vulnerabilities,
                "recommendations": await self._generate_moat_recommendations(moat_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing competitive moats: {str(e)}")
            return {"error": str(e)}
            
    async def _prepare_competitive_features(
        self,
        features: pd.DataFrame,
        company_ids: List[str]
    ) -> pd.DataFrame:
        """Prepare features for competitive analysis."""
        
        try:
            # Filter for relevant companies
            comp_features = features[features['company_id'].isin(company_ids)].copy()
            
            if comp_features.empty:
                return pd.DataFrame()
                
            # Aggregate features by company
            agg_features = comp_features.groupby('company_id').agg({
                'hiring_rate': 'mean',
                'job_posting_count': 'sum',
                'technology_diversity': 'mean',
                'innovation_score': 'mean',
                'talent_quality_score': 'mean',
                'market_presence_score': 'mean',
                'growth_rate': 'mean',
                'funding_amount': 'sum',
                'patent_count': 'sum',
                'github_stars': 'sum',
                'employee_count': 'mean',
                'sentiment_score': 'mean'
            }).fillna(0)
            
            # Add derived features
            agg_features['hiring_efficiency'] = (
                agg_features['hiring_rate'] / (agg_features['employee_count'] + 1)
            )
            agg_features['innovation_per_employee'] = (
                agg_features['innovation_score'] / (agg_features['employee_count'] + 1)
            )
            agg_features['tech_adoption_rate'] = (
                agg_features['technology_diversity'] / agg_features['job_posting_count'].max()
            )
            
            # Normalize features
            normalized_features = pd.DataFrame(
                self.scaler.fit_transform(agg_features),
                columns=agg_features.columns,
                index=agg_features.index
            )
            
            return normalized_features
            
        except Exception as e:
            logger.error(f"Error preparing competitive features: {str(e)}")
            return pd.DataFrame()
            
    async def _create_positioning_map(
        self,
        features: pd.DataFrame,
        company_ids: List[str]
    ) -> Dict[str, Any]:
        """Create competitive positioning map using dimensionality reduction."""
        
        try:
            if len(features) < 2:
                return {"error": "Need at least 2 companies for positioning map"}
                
            # Apply PCA for positioning
            pca_coords = self.pca_model.fit_transform(features)
            
            # Apply t-SNE for alternative view
            tsne_coords = None
            if len(features) >= 3:  # t-SNE needs at least 3 samples
                self.tsne_model = TSNE(n_components=2, random_state=42, perplexity=min(5, len(features)-1))
                tsne_coords = self.tsne_model.fit_transform(features)
                
            positioning_map = {
                "pca_coordinates": {
                    company_id: {"x": float(pca_coords[i, 0]), "y": float(pca_coords[i, 1])}
                    for i, company_id in enumerate(features.index)
                },
                "pca_explained_variance": self.pca_model.explained_variance_ratio_.tolist(),
                "feature_contributions": {
                    "pc1": dict(zip(features.columns, self.pca_model.components_[0])),
                    "pc2": dict(zip(features.columns, self.pca_model.components_[1]))
                }
            }
            
            if tsne_coords is not None:
                positioning_map["tsne_coordinates"] = {
                    company_id: {"x": float(tsne_coords[i, 0]), "y": float(tsne_coords[i, 1])}
                    for i, company_id in enumerate(features.index)
                }
                
            return positioning_map
            
        except Exception as e:
            logger.error(f"Error creating positioning map: {str(e)}")
            return {"error": str(e)}
            
    async def _perform_clustering_analysis(
        self,
        features: pd.DataFrame,
        company_ids: List[str]
    ) -> Dict[str, Any]:
        """Perform clustering analysis to identify competitive groups."""
        
        try:
            if len(features) < 3:
                return {"clusters": {}, "message": "Need at least 3 companies for clustering"}
                
            # Determine optimal number of clusters
            optimal_k = min(5, len(features) // 2)
            self.clustering_model = KMeans(n_clusters=optimal_k, random_state=42)
            
            cluster_labels = self.clustering_model.fit_predict(features)
            
            # Create cluster groups
            clusters = {}
            for i, company_id in enumerate(features.index):
                cluster_id = int(cluster_labels[i])
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(company_id)
                
            # Analyze cluster characteristics
            cluster_characteristics = {}
            for cluster_id, companies in clusters.items():
                cluster_features = features.loc[companies]
                characteristics = {
                    "size": len(companies),
                    "companies": companies,
                    "centroid": cluster_features.mean().to_dict(),
                    "dominant_features": cluster_features.mean().nlargest(3).index.tolist(),
                    "description": await self._describe_cluster(cluster_features)
                }
                cluster_characteristics[cluster_id] = characteristics
                
            return {
                "clusters": clusters,
                "characteristics": cluster_characteristics,
                "silhouette_score": await self._calculate_silhouette_score(features, cluster_labels)
            }
            
        except Exception as e:
            logger.error(f"Error performing clustering analysis: {str(e)}")
            return {"error": str(e)}
            
    async def _perform_swot_analysis(
        self,
        features: pd.DataFrame,
        company_ids: List[str]
    ) -> Dict[str, Any]:
        """Perform SWOT analysis for each company."""
        
        try:
            swot_analysis = {}
            
            for company_id in company_ids:
                if company_id not in features.index:
                    continue
                    
                company_features = features.loc[company_id]
                
                # Identify strengths (top 3 features)
                strengths = company_features.nlargest(3).index.tolist()
                
                # Identify weaknesses (bottom 3 features)
                weaknesses = company_features.nsmallest(3).index.tolist()
                
                # Calculate relative position vs market
                market_avg = features.mean()
                relative_position = company_features - market_avg
                
                # Opportunities (areas where company is below market but market is growing)
                opportunities = await self._identify_opportunities(company_id, relative_position)
                
                # Threats (areas where company is strong but market is declining)
                threats = await self._identify_threats(company_id, relative_position)
                
                swot_analysis[company_id] = {
                    "strengths": [self._feature_to_readable(s) for s in strengths],
                    "weaknesses": [self._feature_to_readable(w) for w in weaknesses],
                    "opportunities": opportunities,
                    "threats": threats,
                    "relative_scores": relative_position.to_dict()
                }
                
            return {"strengths": swot_analysis}
            
        except Exception as e:
            logger.error(f"Error performing SWOT analysis: {str(e)}")
            return {"strengths": {}}
            
    async def _perform_gap_analysis(
        self,
        features: pd.DataFrame,
        company_ids: List[str]
    ) -> Dict[str, Any]:
        """Perform gap analysis to identify improvement opportunities."""
        
        try:
            gap_analysis = {}
            
            # Calculate market leader benchmarks
            market_leaders = features.max()
            market_avg = features.mean()
            
            for company_id in company_ids:
                if company_id not in features.index:
                    continue
                    
                company_features = features.loc[company_id]
                
                # Calculate gaps to market leader
                leader_gaps = market_leaders - company_features
                
                # Calculate gaps to market average
                avg_gaps = market_avg - company_features
                
                # Prioritize gaps (largest gaps with highest market importance)
                prioritized_gaps = leader_gaps.nlargest(5).to_dict()
                
                gap_analysis[company_id] = {
                    "gaps_to_leader": {k: float(v) for k, v in leader_gaps.items()},
                    "gaps_to_average": {k: float(v) for k, v in avg_gaps.items()},
                    "priority_gaps": prioritized_gaps,
                    "improvement_recommendations": await self._generate_improvement_recommendations(
                        company_id, prioritized_gaps
                    )
                }
                
            return gap_analysis
            
        except Exception as e:
            logger.error(f"Error performing gap analysis: {str(e)}")
            return {}
            
    async def _generate_strategic_recommendations(
        self,
        features: pd.DataFrame,
        positioning_map: Dict[str, Any],
        clusters: Dict[str, Any],
        gap_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate strategic recommendations based on competitive analysis."""
        
        try:
            recommendations = {}
            
            for company_id in features.index:
                company_recs = []
                
                # Position-based recommendations
                if "pca_coordinates" in positioning_map:
                    coords = positioning_map["pca_coordinates"].get(company_id, {})
                    if coords:
                        company_recs.extend(
                            await self._get_position_based_recommendations(coords, features.loc[company_id])
                        )
                        
                # Cluster-based recommendations
                if "characteristics" in clusters:
                    for cluster_id, characteristics in clusters["characteristics"].items():
                        if company_id in characteristics.get("companies", []):
                            company_recs.extend(
                                await self._get_cluster_based_recommendations(characteristics)
                            )
                            break
                            
                # Gap-based recommendations
                if company_id in gap_analysis:
                    company_recs.extend(
                        gap_analysis[company_id].get("improvement_recommendations", [])
                    )
                    
                recommendations[company_id] = list(set(company_recs))  # Remove duplicates
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating strategic recommendations: {str(e)}")
            return {}
            
    async def _analyze_market_dynamics(
        self,
        features: pd.DataFrame,
        company_ids: List[str]
    ) -> Dict[str, Any]:
        """Analyze market dynamics and competitive forces."""
        
        try:
            # Calculate market concentration
            market_shares = await self._estimate_market_shares(features)
            hhi_index = sum(share**2 for share in market_shares.values()) * 10000
            
            # Analyze competitive intensity
            competitive_intensity = await self._calculate_competitive_intensity(features)
            
            # Innovation pace
            innovation_pace = features['innovation_score'].mean()
            
            # Market growth rate
            growth_rate = features['growth_rate'].mean()
            
            # Barrier to entry analysis
            barriers_to_entry = await self._analyze_barriers_to_entry(features)
            
            # Supplier/buyer power
            market_power = await self._analyze_market_power(features)
            
            return {
                "market_concentration": {
                    "hhi_index": float(hhi_index),
                    "interpretation": self._interpret_hhi(hhi_index),
                    "market_shares": market_shares
                },
                "competitive_intensity": competitive_intensity,
                "innovation_pace": float(innovation_pace),
                "market_growth_rate": float(growth_rate),
                "barriers_to_entry": barriers_to_entry,
                "market_power": market_power,
                "market_attractiveness": await self._calculate_market_attractiveness(
                    hhi_index, competitive_intensity, innovation_pace, growth_rate
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market dynamics: {str(e)}")
            return {}
            
    # Helper methods for specific analyses
    
    async def _get_company_features(self, company_id: str) -> pd.DataFrame:
        """Get comprehensive features for a specific company."""
        # Implementation would fetch from database
        return pd.DataFrame()
        
    async def _get_market_data(self, industry: Optional[str], region: Optional[str]) -> pd.DataFrame:
        """Get market data filtered by industry and region."""
        # Implementation would fetch from database
        return pd.DataFrame()
        
    async def _calculate_leadership_scores(
        self,
        market_data: pd.DataFrame,
        metrics: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Calculate leadership scores for companies."""
        # Implementation details...
        return []
        
    async def _describe_cluster(self, cluster_features: pd.DataFrame) -> str:
        """Generate description for a cluster."""
        dominant_features = cluster_features.mean().nlargest(3)
        description = f"Companies strong in {', '.join(dominant_features.index[:2])}"
        return description
        
    async def _calculate_silhouette_score(self, features: pd.DataFrame, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality."""
        try:
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(features, labels))
        except:
            return 0.0
            
    def _feature_to_readable(self, feature_name: str) -> str:
        """Convert feature name to readable format."""
        readable_names = {
            'hiring_rate': 'Hiring Rate',
            'innovation_score': 'Innovation Capability',
            'talent_quality_score': 'Talent Quality',
            'market_presence_score': 'Market Presence',
            'technology_diversity': 'Technology Diversity',
            'growth_rate': 'Growth Rate',
            'funding_amount': 'Funding Amount',
            'patent_count': 'Patent Portfolio',
            'github_stars': 'Developer Mindshare'
        }
        return readable_names.get(feature_name, feature_name.replace('_', ' ').title())
        
    async def _identify_opportunities(self, company_id: str, relative_position: pd.Series) -> List[str]:
        """Identify opportunities based on relative market position."""
        opportunities = []
        
        # Areas where company is below market average but market is growing
        weak_areas = relative_position.nsmallest(3)
        
        for area in weak_areas.index:
            if 'growth' in area.lower() or 'innovation' in area.lower():
                opportunities.append(f"Invest in {self._feature_to_readable(area)}")
                
        return opportunities[:3]
        
    async def _identify_threats(self, company_id: str, relative_position: pd.Series) -> List[str]:
        """Identify threats based on market position."""
        threats = []
        
        # Generic threats for now
        threats.extend([
            "Increasing competition in core markets",
            "Technology disruption risk",
            "Talent acquisition challenges"
        ])
        
        return threats[:3]
        
    async def _generate_improvement_recommendations(
        self,
        company_id: str,
        priority_gaps: Dict[str, float]
    ) -> List[str]:
        """Generate specific improvement recommendations."""
        recommendations = []
        
        for gap_area, gap_size in list(priority_gaps.items())[:3]:
            if gap_size > 0:  # Only recommend if there's actually a gap
                readable_area = self._feature_to_readable(gap_area)
                recommendations.append(f"Focus on improving {readable_area}")
                
        return recommendations
        
    async def _get_position_based_recommendations(
        self,
        coords: Dict[str, float],
        company_features: pd.Series
    ) -> List[str]:
        """Get recommendations based on position in competitive map."""
        recommendations = []
        
        # Simplified position-based logic
        if coords.get("x", 0) < 0:
            recommendations.append("Consider market expansion strategies")
        if coords.get("y", 0) < 0:
            recommendations.append("Focus on operational efficiency")
            
        return recommendations
        
    async def _get_cluster_based_recommendations(self, characteristics: Dict[str, Any]) -> List[str]:
        """Get recommendations based on cluster membership."""
        recommendations = []
        
        dominant_features = characteristics.get("dominant_features", [])
        if "innovation_score" in dominant_features:
            recommendations.append("Leverage innovation capabilities for market differentiation")
        if "hiring_rate" in dominant_features:
            recommendations.append("Capitalize on talent acquisition advantages")
            
        return recommendations
        
    async def _estimate_market_shares(self, features: pd.DataFrame) -> Dict[str, float]:
        """Estimate market shares based on various metrics."""
        # Simplified market share estimation
        total_size = features['market_presence_score'].sum()
        market_shares = {}
        
        for company_id in features.index:
            share = features.loc[company_id, 'market_presence_score'] / total_size if total_size > 0 else 0
            market_shares[company_id] = float(share)
            
        return market_shares
        
    async def _calculate_competitive_intensity(self, features: pd.DataFrame) -> Dict[str, float]:
        """Calculate competitive intensity metrics."""
        
        # Number of competitors
        num_competitors = len(features)
        
        # Feature variance (higher variance = more differentiated market)
        feature_variance = features.var().mean()
        
        # Competition score
        competition_score = num_competitors * (1 - feature_variance)
        
        return {
            "number_of_competitors": num_competitors,
            "market_differentiation": float(feature_variance),
            "intensity_score": float(competition_score)
        }
        
    def _interpret_hhi(self, hhi: float) -> str:
        """Interpret Herfindahl-Hirschman Index."""
        if hhi < 1500:
            return "Highly competitive market"
        elif hhi < 2500:
            return "Moderately concentrated market"
        else:
            return "Highly concentrated market"
            
    async def _analyze_barriers_to_entry(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze barriers to entry in the market."""
        
        # High innovation requirements
        innovation_barrier = features['innovation_score'].mean()
        
        # Capital requirements (funding amounts)
        capital_barrier = features['funding_amount'].mean()
        
        # Talent requirements
        talent_barrier = features['talent_quality_score'].mean()
        
        return {
            "innovation_barrier": float(innovation_barrier),
            "capital_barrier": float(capital_barrier),
            "talent_barrier": float(talent_barrier),
            "overall_barrier_height": float((innovation_barrier + capital_barrier + talent_barrier) / 3)
        }
        
    async def _analyze_market_power(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze buyer and supplier power in the market."""
        
        # Simplified analysis
        return {
            "buyer_power": "Medium",  # Would analyze customer concentration
            "supplier_power": "Low",  # Would analyze talent market conditions
            "substitution_threat": "High"  # Tech markets typically have high substitution risk
        }
        
    async def _calculate_market_attractiveness(
        self,
        hhi: float,
        competitive_intensity: Dict[str, float],
        innovation_pace: float,
        growth_rate: float
    ) -> Dict[str, Any]:
        """Calculate overall market attractiveness score."""
        
        # Normalize scores
        concentration_score = max(0, (2500 - hhi) / 2500)  # Lower concentration = more attractive
        competition_score = max(0, 1 - competitive_intensity.get("intensity_score", 0) / 10)
        innovation_score = min(1, innovation_pace)
        growth_score = min(1, max(0, growth_rate))
        
        overall_score = (concentration_score + competition_score + innovation_score + growth_score) / 4
        
        return {
            "overall_score": float(overall_score),
            "concentration_score": float(concentration_score),
            "competition_score": float(competition_score),
            "innovation_score": float(innovation_score),
            "growth_score": float(growth_score),
            "interpretation": self._interpret_attractiveness(overall_score)
        }
        
    def _interpret_attractiveness(self, score: float) -> str:
        """Interpret market attractiveness score."""
        if score > 0.8:
            return "Highly attractive market"
        elif score > 0.6:
            return "Moderately attractive market"
        elif score > 0.4:
            return "Average market attractiveness"
        else:
            return "Low market attractiveness"
            
    # Additional methods for comprehensive competitive analysis would be implemented here
    # Including moat analysis, technology overlap, talent competition, etc.