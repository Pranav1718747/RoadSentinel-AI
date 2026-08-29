from celery import shared_task
from processing.services.pipeline_service import run_pipeline_for_job

@shared_task(bind=True)
def process_video_pipeline(self, job_id):
    """
    Celery task wrapper executing the RoadSentinel pipeline for a job ID.
    Passes self as celery_task for real-time progress state updates.
    """
    print(f"[Celery Worker] Starting pipeline execution for ProcessingJob #{job_id}")
    result = run_pipeline_for_job(job_id, celery_task=self)
    print(f"[Celery Worker] Completed pipeline execution for ProcessingJob #{job_id} (Status: {result})")
    return result
