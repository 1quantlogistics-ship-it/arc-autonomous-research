import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

from config import get_settings

# Configuration
CONTROL_PLANE_URL = 'http://localhost:8002'
LLM_ENDPOINT = 'http://localhost:8000'

st.set_page_config(
    page_title='ARC Dashboard',
    page_icon='\U0001F916',
    layout='wide'
)

# Sidebar
st.sidebar.title('ARC Control Panel')
st.sidebar.markdown('---')

# Get system status
def get_status():
    try:
        response = requests.get(f'{CONTROL_PLANE_URL}/status', timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.sidebar.error(f'Control Plane unreachable: {e}')
        return None

status = get_status()

if status:
    # Mode display
    mode = status.get('mode', 'UNKNOWN')
    mode_colors = {'SEMI': 'orange', 'AUTO': 'blue', 'FULL': 'red'}
    st.sidebar.markdown(f'### Mode: :{mode_colors.get(mode, "gray")}[{mode}]')
    
    # System info
    st.sidebar.metric('ARC Version', status.get('arc_version', 'N/A'))
    st.sidebar.metric('Status', status.get('status', 'N/A'))
    st.sidebar.metric('Current Cycle', status.get('current_cycle', 0))
    st.sidebar.metric('Active Experiments', len(status.get('active_experiments', [])))
    
    # Mode switcher
    st.sidebar.markdown('---')
    st.sidebar.subheader('Change Mode')
    new_mode = st.sidebar.selectbox('Select Mode', ['SEMI', 'AUTO', 'FULL'])
    if st.sidebar.button('Apply Mode'):
        try:
            response = requests.post(f'{CONTROL_PLANE_URL}/mode?mode={new_mode}')
            if response.status_code == 200:
                st.sidebar.success(f'Mode changed to {new_mode}')
                time.sleep(1)
                st.rerun()
            else:
                st.sidebar.error('Mode change failed')
        except Exception as e:
            st.sidebar.error(f'Error: {e}')

# Main dashboard
st.title('ARC - Autonomous Research Collective')
st.markdown('### Multi-Agent LLM Research Platform')

if not status:
    st.error('Cannot connect to Control Plane. Please ensure the service is running.')
    st.code('python /workspace/arc/api/control_plane.py')
    st.stop()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    'Overview', 'Memory', 'Experiments', 'Logs', 'Execute', 'Agents', 'Supervisor', 'Insights'
])

with tab1:
    st.header('System Overview')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric('Operating Mode', mode)
    with col2:
        st.metric('Current Objective', status.get('current_objective', 'N/A'))
    with col3:
        last_cycle = status.get('last_cycle')
        if last_cycle:
            st.metric('Last Cycle', datetime.fromisoformat(last_cycle).strftime('%Y-%m-%d %H:%M'))
        else:
            st.metric('Last Cycle', 'Never')
    with col4:
        st.metric('Cycle ID', status.get('current_cycle', 0))
    
    st.markdown('---')
    
    # Active experiments
    st.subheader('Active Experiments')
    active_exps = status.get('active_experiments', [])
    if active_exps:
        df = pd.DataFrame(active_exps)
        st.dataframe(df, use_container_width=True)
    else:
        st.info('No active experiments')

with tab2:
    st.header('Protocol Memory')

    settings = get_settings()
    memory_files = ['directive.json', 'history_summary.json', 'constraints.json', 'system_state.json']

    for fname in memory_files:
        with st.expander(f'\U0001F4C4 {fname}'):
            try:
                with open(f'{settings.memory_dir}/{fname}', 'r') as f:
                    data = json.load(f)
                st.json(data)
            except Exception as e:
                st.error(f'Error loading {fname}: {e}')

with tab3:
    st.header('Experiments')

    try:
        import os
        settings = get_settings()
        exp_dir = str(settings.experiments_dir)
        if os.path.exists(exp_dir):
            experiments = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
            
            if experiments:
                selected_exp = st.selectbox('Select Experiment', experiments)
                
                if selected_exp:
                    exp_path = os.path.join(exp_dir, selected_exp)
                    
                    # Load experiment metadata
                    meta_path = os.path.join(exp_path, 'metadata.json')
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r') as f:
                            metadata = json.load(f)
                        st.json(metadata)
                    
                    # Load results
                    results_path = os.path.join(exp_path, 'results.json')
                    if os.path.exists(results_path):
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                        
                        st.subheader('Results')
                        st.json(results)
                        
                        # Plot metrics if available
                        if 'metrics' in results:
                            metrics = results['metrics']
                            fig = go.Figure(data=[
                                go.Bar(x=list(metrics.keys()), y=list(metrics.values()))
                            ])
                            fig.update_layout(title='Experiment Metrics')
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No experiments found')
        else:
            st.info('Experiments directory not found')
    except Exception as e:
        st.error(f'Error loading experiments: {e}')

with tab4:
    st.header('System Logs')
    
    log_type = st.selectbox('Log Type', ['Control Plane', 'Execution Logs', 'Training Logs'])
    
    try:
        if log_type == 'Control Plane':
            log_path = '/workspace/arc/logs/control_plane.log'
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    last_n = st.slider('Show last N lines', 10, 1000, 100)
                    st.text_area('Log Output', ''.join(lines[-last_n:]), height=400)
            else:
                st.info('No logs available')
        elif log_type == 'Execution Logs':
            log_dir = '/workspace/arc/logs'
            exec_logs = [f for f in os.listdir(log_dir) if f.startswith('exec_cycle_')]
            if exec_logs:
                selected_log = st.selectbox('Select Log File', exec_logs)
                with open(os.path.join(log_dir, selected_log), 'r') as f:
                    st.text_area('Log Output', f.read(), height=400)
            else:
                st.info('No execution logs available')
        else:
            st.info('Training logs not yet implemented')
    except Exception as e:
        st.error(f'Error loading logs: {e}')

with tab5:
    st.header('Execute Commands')
    
    st.warning('This tab allows executing commands through the ARC Control Plane. Commands are validated against the allowlist.')
    
    with st.form('exec_form'):
        role = st.selectbox('Role', ['Director', 'Architect', 'Critic', 'Historian', 'Executor'])
        cycle_id = st.number_input('Cycle ID', min_value=0, value=status.get('current_cycle', 0))
        command = st.text_input('Command')
        requires_approval = st.checkbox('Requires Approval', value=True)
        
        submit = st.form_submit_button('Execute')
        
        if submit and command:
            try:
                payload = {
                    'command': command,
                    'role': role,
                    'cycle_id': int(cycle_id),
                    'requires_approval': requires_approval
                }
                
                response = requests.post(f'{CONTROL_PLANE_URL}/exec', json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get('status') == 'pending_approval':
                        st.warning(result.get('message'))
                    else:
                        st.success('Command executed')
                        st.json(result)
                else:
                    st.error(f'Execution failed: {response.text}')
            except Exception as e:
                st.error(f'Error: {e}')

with tab6:
    st.header('Agent Management')

    # Try to load real data first, then fall back to mock
    agent_status = []
    data_source = "mock"

    try:
        from api.dashboard_adapter import get_dashboard_adapter
        from api.multi_agent_orchestrator import MultiAgentOrchestrator

        # Try to get real orchestrator state
        try:
            settings = get_settings()
            orchestrator = MultiAgentOrchestrator(
                memory_path=str(settings.memory_dir),
                offline_mode=True
            )
            adapter = get_dashboard_adapter(str(settings.memory_dir))
            agent_status = adapter.get_agent_status(orchestrator.registry)
            data_source = "real"
            st.success('✓ Connected to orchestrator - showing real agent data')
        except Exception as e:
            st.info(f'Using demo data (orchestrator not available: {str(e)[:50]}...)')
            from api.mock_data import get_mock_data
            mock_data = get_mock_data()
            agent_status = mock_data.get("agent_status", [])

        if agent_status:

            # Agent status table
            st.subheader('Agent Registry Status')
            df_agents = pd.DataFrame(agent_status)

            # Format table
            display_cols = ['agent_id', 'role', 'model', 'state', 'voting_weight', 'healthy']
            st.dataframe(df_agents[display_cols], use_container_width=True)

            # Agent activity timeline
            st.subheader('Recent Agent Activity')
            active_agents = [a for a in agent_status if a['state'] in ['active', 'busy']]

            for agent in active_agents[:5]:
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.text(f"{agent['agent_id']} ({agent['role']})")
                with col2:
                    st.text(f"Last activity: {agent['last_activity'][:19]}")
                with col3:
                    status_color = "green" if agent['state'] == 'active' else "orange"
                    st.markdown(f":{status_color}[{agent['state']}]")

            # Agent performance metrics
            st.subheader('Agent Performance Metrics')

            for agent in agent_status[:3]:
                with st.expander(f"{agent['agent_id']} - {agent['role']}"):
                    metrics = agent['metrics']

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric('Total Tasks', metrics['total_tasks'])
                    with col2:
                        success_rate = (metrics['successful_tasks'] / metrics['total_tasks'] * 100) if metrics['total_tasks'] > 0 else 0
                        st.metric('Success Rate', f"{success_rate:.1f}%")
                    with col3:
                        st.metric('Avg Response Time', f"{metrics['avg_response_time_ms']:.0f}ms")

                    col4, col5 = st.columns(2)
                    with col4:
                        st.metric('Total Votes', metrics['total_votes'])
                    with col5:
                        st.metric('Vote Agreement', f"{metrics['vote_agreement_rate']*100:.1f}%")
        else:
            st.warning('No agent data available. Start agents to see status.')
    except Exception as e:
        st.error(f'Error loading agent data: {e}')

with tab7:
    st.header('Supervisor Oversight')

    try:
        from api.dashboard_adapter import get_dashboard_adapter

        # Try to load real data first
        settings = get_settings()
        adapter = get_dashboard_adapter(str(settings.memory_dir))
        supervisor_decisions = adapter.get_supervisor_decisions(limit=100)
        risk_distribution = adapter.get_risk_distribution()

        if not supervisor_decisions:
            # Fall back to mock data
            st.info('Demo Mode: No supervisor decisions yet - showing mock data')
            from api.mock_data import get_mock_data
            mock_data = get_mock_data()
            supervisor_decisions = mock_data.get("supervisor_decisions", [])
            risk_distribution = mock_data.get("risk_distribution", {})
        else:
            st.success(f'✓ Showing {len(supervisor_decisions)} real supervisor decisions')

        if supervisor_decisions:

            # Recent supervisor decisions
            st.subheader('Recent Supervisor Decisions')

            # Decision summary
            col1, col2, col3, col4 = st.columns(4)
            decision_counts = {'approve': 0, 'reject': 0, 'revise': 0, 'override': 0}
            for d in supervisor_decisions:
                dec = d['decision']
                if dec in decision_counts:
                    decision_counts[dec] += 1

            with col1:
                st.metric('Approved', decision_counts['approve'])
            with col2:
                st.metric('Rejected', decision_counts['reject'])
            with col3:
                st.metric('Revisions Requested', decision_counts['revise'])
            with col4:
                st.metric('Overrides', decision_counts['override'])

            # Risk level distribution
            st.subheader('Risk Level Distribution')

            if risk_distribution:
                fig = go.Figure(data=[go.Pie(
                    labels=list(risk_distribution.keys()),
                    values=list(risk_distribution.values()),
                    marker=dict(colors=['#00cc96', '#ffa500', '#ff6347', '#dc143c'])
                )])
                fig.update_layout(title='Proposal Risk Levels')
                st.plotly_chart(fig, use_container_width=True)

            # Decision details table
            st.subheader('Decision History')
            df_decisions = pd.DataFrame(supervisor_decisions)
            display_cols = ['proposal_id', 'decision', 'risk_assessment', 'override_consensus', 'confidence']
            st.dataframe(df_decisions[display_cols].head(10), use_container_width=True)

            # Override history
            st.subheader('Supervisor Overrides')
            overrides = [d for d in supervisor_decisions if d['override_consensus']]

            if overrides:
                for override in overrides[:5]:
                    with st.expander(f"{override['proposal_id']} - {override['decision'].upper()}"):
                        st.markdown(f"**Risk**: {override['risk_assessment']}")
                        st.markdown(f"**Reasoning**: {override['reasoning']}")
                        st.markdown(f"**Confidence**: {override['confidence']:.2f}")
                        if override['constraints_violated']:
                            st.markdown(f"**Constraints Violated**: {', '.join(override['constraints_violated'])}")
            else:
                st.info('No supervisor overrides in recent history')
        else:
            st.warning('No supervisor decision data available')
    except Exception as e:
        st.error(f'Error loading supervisor data: {e}')

with tab8:
    st.header('Multi-Agent Insights')

    try:
        from api.dashboard_adapter import get_dashboard_adapter

        # Try to load real data first
        settings = get_settings()
        adapter = get_dashboard_adapter(str(settings.memory_dir))
        consensus_metrics = adapter.get_consensus_metrics()
        voting_patterns = adapter.get_voting_patterns()
        proposal_quality = adapter.get_proposal_quality_trends()

        if consensus_metrics.get('total_votes_conducted', 0) == 0:
            # Fall back to mock data
            st.info('Demo Mode: No voting data yet - showing mock insights')
            from api.mock_data import get_mock_data
            mock_data = get_mock_data()
            consensus_metrics = mock_data.get("consensus_metrics", {})
            voting_patterns = mock_data.get("voting_patterns", {})
            proposal_quality = mock_data.get("proposal_quality_trends", [])
        else:
            st.success(f'✓ Showing real multi-agent insights ({consensus_metrics["total_votes_conducted"]} votes)')

        if consensus_metrics:

            # Consensus metrics
            st.subheader('Consensus Quality Metrics')

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric('Total Votes', consensus_metrics.get('total_votes_conducted', 0))
            with col2:
                st.metric('Consensus Rate', f"{consensus_metrics.get('consensus_rate', 0)*100:.1f}%")
            with col3:
                st.metric('Avg Confidence', f"{consensus_metrics.get('avg_confidence', 0)*100:.1f}%")
            with col4:
                st.metric('Controversial Rate', f"{consensus_metrics.get('controversial_rate', 0)*100:.1f}%")

            # Decision breakdown
            st.subheader('Decision Distribution')

            decision_breakdown = consensus_metrics.get('decision_breakdown', {})
            if decision_breakdown:
                fig = go.Figure(data=[go.Bar(
                    x=list(decision_breakdown.keys()),
                    y=list(decision_breakdown.values()),
                    marker=dict(color=['#00cc96', '#ff6347', '#ffa500'])
                )])
                fig.update_layout(title='Consensus Decisions', xaxis_title='Decision', yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)

            # Agent voting patterns heatmap
            st.subheader('Agent Voting Agreement Patterns')

            if voting_patterns:
                st.info('Heatmap shows agreement rates between agents (higher = more agreement)')

                # Create simple agreement summary
                st.markdown('**High Agreement Pairs** (>85% agreement):')
                high_agreement = []
                for agent1, patterns in voting_patterns.items():
                    for agent2, agreement in patterns.items():
                        if agent1 < agent2 and agreement > 0.85:
                            high_agreement.append(f"- {agent1} ↔ {agent2}: {agreement*100:.1f}%")

                if high_agreement:
                    for pair in high_agreement[:5]:
                        st.text(pair)
                else:
                    st.text('No high agreement pairs found')

            # Proposal quality trends
            st.subheader('Proposal Quality Over Time')

            if proposal_quality:
                df_quality = pd.DataFrame(proposal_quality)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_quality['cycle_id'],
                    y=df_quality['avg_proposal_quality'],
                    mode='lines+markers',
                    name='Proposal Quality',
                    line=dict(color='#00cc96')
                ))
                fig.add_trace(go.Scatter(
                    x=df_quality['cycle_id'],
                    y=df_quality['consensus_score'],
                    mode='lines+markers',
                    name='Consensus Score',
                    line=dict(color='#636efa')
                ))
                fig.update_layout(
                    title='Proposal Quality and Consensus Trends',
                    xaxis_title='Cycle ID',
                    yaxis_title='Score',
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True)

                # Approval/rejection trends
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=df_quality['cycle_id'],
                    y=df_quality['proposals_approved'],
                    name='Approved',
                    marker=dict(color='#00cc96')
                ))
                fig2.add_trace(go.Bar(
                    x=df_quality['cycle_id'],
                    y=df_quality['proposals_rejected'],
                    name='Rejected',
                    marker=dict(color='#ff6347')
                ))
                fig2.update_layout(
                    title='Proposal Approval Trends',
                    xaxis_title='Cycle ID',
                    yaxis_title='Count',
                    barmode='stack'
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning('No consensus insights data available')
    except Exception as e:
        st.error(f'Error loading insights data: {e}')

# Footer
st.markdown('---')
st.markdown('ARC v1.1.0-alpha | Phase D: Multi-Agent Architecture | Autonomous Research Collective')
