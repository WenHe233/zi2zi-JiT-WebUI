def periodic_or_final(epoch: int, total_epochs: int, frequency: int) -> bool:
    completed_epochs = epoch + 1
    return completed_epochs >= total_epochs or (
        frequency > 0 and completed_epochs % frequency == 0
    )
